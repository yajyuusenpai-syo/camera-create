"""Schedule manifest-listed videos across isolated metric-camera GPU workers."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from .artifacts import export_camera_json_v2
from .config import ModelPaths
from .distributed import (
    DistributedLayout,
    TaskLease,
    assign_node_tasks,
    ensure_shared_manifest,
    validate_run_id,
)
from .pipeline import CameraCreatePipeline, PipelineOptions
from .stage_cache import video_identity
from .worker_runner import (
    DepthServiceEndpoint,
    DepthServiceProcess,
    start_depth_service,
)

DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchOptions:
    """Serializable settings shared by all statically assigned batch workers."""

    input_manifest: Path
    checkpoint_root: Path
    model_paths: ModelPaths
    pipeline_options: PipelineOptions
    gpu_ids: tuple[int, ...]
    workers_per_gpu: int = 1
    depth_services_per_gpu: int = 1
    node_rank: int = 0
    num_nodes: int = 1
    run_id: str | None = None
    launcher_num_processes: int | None = None
    main_process_ip: str | None = None
    main_process_port: int | None = None
    lease_timeout_seconds: float = 900.0
    target_fps: float = 24.0
    max_frames: int = 241
    max_video_seconds: float = 10.06
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS
    ffmpeg_command: str = "ffmpeg"
    overwrite: bool = False
    keep_stage_cache: bool = False


def discover_videos(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Use os.walk to return a stable recursive list of supported video files."""
    root = root.resolve()
    normalized = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
    }
    videos: list[Path] = []
    checkpoint_name = ".camera_create_ckpt"
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name != checkpoint_name)
        for filename in sorted(files):
            candidate = Path(directory) / filename
            if candidate.suffix.lower() in normalized:
                videos.append(candidate.resolve())
    return sorted(videos, key=lambda path: path.relative_to(root).as_posix())


def load_video_manifest(path: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Load an ordered TXT/JSON video-path shard and validate every source file."""
    manifest = path.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {manifest}")
    normalized_extensions = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
    }
    suffix = manifest.suffix.lower()
    if suffix == ".txt":
        raw_entries: list[Any] = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif suffix == ".json":
        document = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if isinstance(document, dict):
            for key in ("videos", "paths"):
                if key in document:
                    document = document[key]
                    break
        if not isinstance(document, list):
            raise ValueError(
                "JSON input manifest must be an array, or an object containing "
                "a 'videos' or 'paths' array"
            )
        raw_entries = document
    else:
        raise ValueError("--input must be a .txt or .json video-path manifest")

    videos: list[Path] = []
    seen: set[Path] = set()
    for index, entry in enumerate(raw_entries, start=1):
        if isinstance(entry, dict):
            entry = entry.get("path", entry.get("video_path"))
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Invalid video path at manifest item {index}: {entry!r}")
        candidate = Path(entry.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        candidate = candidate.resolve()
        if candidate in seen:
            raise ValueError(f"Duplicate video path in input manifest: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Video listed at manifest item {index} does not exist: {candidate}"
            )
        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Unsupported video extension at manifest item {index}: {candidate}"
            )
        seen.add(candidate)
        videos.append(candidate)
    return videos


def _task_name(video: Path) -> str:
    """Return the stable absolute identity stored in manifests and checkpoints."""
    return video.resolve().as_posix()


def _start_depth_service_group(
    model: str,
    gpu_ids: tuple[int, ...],
    options: BatchOptions,
    service_root: Path,
) -> tuple[list[DepthServiceProcess], dict[tuple[int, int], DepthServiceEndpoint]]:
    """Load model replicas in waves across GPUs and expose their endpoints."""
    LOG.info(
        "Loading %d persistent %s services per GPU on GPUs %s",
        options.depth_services_per_gpu,
        model,
        ",".join(str(gpu_id) for gpu_id in gpu_ids),
    )

    def launch(gpu_id: int, replica_id: int) -> DepthServiceProcess:
        common = {
            "max_side": options.pipeline_options.max_inference_side,
            "disable_cudnn": options.pipeline_options.disable_cudnn,
            "disable_sdp": options.pipeline_options.disable_sdp,
        }
        if model == "Pi3X":
            return start_depth_service(
                model,
                options.pipeline_options.pi3x_python,
                options.model_paths.pi3x,
                gpu_id,
                service_root / f"pi3x_gpu_{gpu_id}_replica_{replica_id}.json",
                chunk=options.pipeline_options.pi3x_chunk,
                stride=options.pipeline_options.pi3x_stride,
                **common,
            )
        return start_depth_service(
            model,
            options.pipeline_options.moge3_python,
            options.model_paths.moge3,
            gpu_id,
            service_root / f"moge3_gpu_{gpu_id}_replica_{replica_id}.json",
            refine_steps=options.pipeline_options.moge3_refine_steps,
            use_fp16=options.pipeline_options.moge3_fp16,
            **common,
        )

    completed: dict[tuple[int, int], DepthServiceProcess] = {}
    failures: list[BaseException] = []
    # One replica wave reads the checkpoint concurrently across all GPUs. Starting
    # every GPU and replica together can overwhelm a shared/FUSE model filesystem.
    for replica_id in range(options.depth_services_per_gpu):
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(launch, gpu_id, replica_id): gpu_id
                for gpu_id in gpu_ids
            }
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    completed[(gpu_id, replica_id)] = future.result()
                    LOG.info(
                        "Persistent %s service replica %d/%d ready on GPU %d",
                        model,
                        replica_id + 1,
                        options.depth_services_per_gpu,
                        gpu_id,
                    )
                except BaseException as error:  # noqa: BLE001 - clean whole group
                    failures.append(error)
        if failures:
            break
    if failures:
        for service in completed.values():
            service.stop()
        raise RuntimeError(f"Failed to load persistent {model} service group") from failures[0]
    keys = [
        (gpu_id, replica_id)
        for replica_id in range(options.depth_services_per_gpu)
        for gpu_id in gpu_ids
    ]
    ordered = [completed[key] for key in keys]
    return ordered, {key: completed[key].endpoint for key in keys}


def camera_json_path(video: Path) -> Path:
    """Place one camera JSON beside its source using the extension-free stem."""
    return video.parent / f"cam_{video.stem}.json"


def camera_artifact_dir(video: Path) -> Path:
    """Place NPY/report artifacts beside the source without stem collisions."""
    return video.parent / f"{video.name}.camera"


def video_lease_path(video: Path) -> Path:
    """Return a manifest-independent lease beside the source video's directory."""
    output_identity = camera_json_path(video.resolve()).as_posix()
    digest = hashlib.sha256(output_identity.encode("utf-8")).hexdigest()[:32]
    return (
        video.resolve().parent
        / ".camera_create_ckpt"
        / "video_leases"
        / f"{digest}.lease"
    )


def validate_unique_camera_outputs(videos: list[Path]) -> None:
    """Reject source files that would map to the same extension-free JSON name."""
    grouped: dict[Path, list[Path]] = {}
    for video in videos:
        grouped.setdefault(camera_json_path(video), []).append(video)
    collisions = {
        output: sources for output, sources in grouped.items() if len(sources) > 1
    }
    if collisions:
        detail = "; ".join(
            f"{output}: {', '.join(str(source) for source in sources)}"
            for output, sources in sorted(collisions.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(
            "Multiple videos would produce the same cam_<stem>.json output: " + detail
        )


def validate_existing_output_owners(videos: list[Path]) -> None:
    """Refuse to overwrite a camera JSON that identifies a different source video."""
    for video in videos:
        output = camera_json_path(video)
        if not output.is_file():
            continue
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        owner = data.get("video_name") if isinstance(data, dict) else None
        if isinstance(owner, str) and owner != video.name:
            raise ValueError(
                f"Output collision: {output} belongs to {owner}, not {video.name}"
            )


def valid_existing_output(
    video: Path,
    target_fps: float | None = None,
    max_frames: int | None = None,
    max_video_seconds: float | None = None,
) -> bool:
    """Recognize only complete metric format-v2 outputs as resumable successes."""
    output = camera_json_path(video)
    if not output.is_file():
        return False
    try:
        data = json.loads(output.read_text(encoding="utf-8"))
        frames = data.get("frames")
        valid = bool(
            data.get("format_version") == 2
            and data.get("video_name") == video.name
            and data.get("is_metric") is True
            and isinstance(frames, list)
            and data.get("frame_count") == len(frames)
            and data.get("frame_count", 0) > 0
        )
        if target_fps is not None:
            valid &= abs(float(data.get("target_fps", -1)) - target_fps) < 1e-6
        if max_frames is not None:
            valid &= data.get("max_frames") == max_frames
            valid &= 0 < int(data.get("frame_count", 0)) <= max_frames
        if max_video_seconds is not None:
            valid &= (
                abs(float(data.get("max_video_seconds", -1)) - max_video_seconds) < 1e-6
            )
        return valid
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def assign_tasks(videos: list[Path], worker_count: int) -> list[list[Path]]:
    """Preassign a deterministic round-robin task list to every worker."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    return [videos[index::worker_count] for index in range(worker_count)]


def depth_service_replica_id(
    local_worker_id: int, workers_per_gpu: int, services_per_gpu: int
) -> int:
    """Bind a local camera worker to one same-GPU model replica."""
    gpu_worker_id = local_worker_id % workers_per_gpu
    return gpu_worker_id % services_per_gpu


def _manifest_payload(
    videos: list[Path], options: BatchOptions, layout: DistributedLayout
) -> dict[str, Any]:
    """Build the immutable task/config contract every distributed node must share."""
    return {
        "format_version": 1,
        "videos": [
            {
                "path": _task_name(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in videos
        ],
        "num_nodes": layout.num_nodes,
        "local_worker_count": layout.local_worker_count,
        "global_worker_count": layout.global_worker_count,
        "launcher_num_processes": options.launcher_num_processes,
        "main_process_ip": options.main_process_ip,
        "main_process_port": options.main_process_port,
        "gpu_ids": list(options.gpu_ids),
        "workers_per_gpu": options.workers_per_gpu,
        "depth_services_per_gpu": options.depth_services_per_gpu,
        "target_fps": options.target_fps,
        "max_frames": options.max_frames,
        "max_video_seconds": options.max_video_seconds,
        "extensions": list(options.extensions),
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "disable_cudnn": options.pipeline_options.disable_cudnn,
        "disable_sdp": options.pipeline_options.disable_sdp,
        "persistent_depth_services": options.pipeline_options.persistent_depth_services,
        "model_paths": {
            "pi3x": str(options.model_paths.pi3x),
            "moge3": str(options.model_paths.moge3),
            "vipe": str(options.model_paths.vipe),
        },
    }


def _probe_video(path: Path) -> tuple[float, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Video reports an invalid FPS: {path}")
    return fps, frames, frames / fps if frames > 0 else 0.0


def prepare_video(
    source: Path,
    directory: Path,
    target_fps: float,
    max_frames: int,
    max_seconds: float,
    ffmpeg_command: str,
) -> tuple[Path, float, float]:
    """Create a bounded constant-FPS processing copy and return its FPS metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    source_fps, _, _ = _probe_video(source)
    processed = directory / f"{source.stem}.normalized.mp4"
    marker = directory / "normalized.json"
    expected = {
        "source": video_identity(source),
        "target_fps": target_fps,
        "max_frames": max_frames,
        "max_seconds": max_seconds,
    }
    if processed.is_file() and marker.is_file():
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
            processed_fps, _, _ = _probe_video(processed)
            if state == expected and abs(processed_fps - target_fps) < 1e-3:
                return processed, source_fps, max_seconds
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    executable = shutil.which(ffmpeg_command)
    if executable is None:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_command}")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        str(max_seconds),
        "-vf",
        f"fps={target_fps}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(max_frames),
        str(processed),
    ]
    subprocess.run(command, check=True)
    _probe_video(processed)
    _atomic_checkpoint(marker, expected)
    return processed, source_fps, max_seconds


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _write_batch_summary(
    run_root: Path, node_rank: int, num_nodes: int, result: dict[str, Any]
) -> None:
    """Write a collision-free node summary and preserve the single-node filename."""
    _atomic_checkpoint(run_root / f"summary_node_{node_rank:03d}.json", result)
    if num_nodes == 1:
        _atomic_checkpoint(run_root / "summary.json", result)


def _run_key(videos: list[Path], options: BatchOptions) -> str:
    stable = {
        "videos": [
            {
                "path": _task_name(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in videos
        ],
        "gpu_ids": options.gpu_ids,
        "workers_per_gpu": options.workers_per_gpu,
        "depth_services_per_gpu": options.depth_services_per_gpu,
        "num_nodes": options.num_nodes,
        "target_fps": options.target_fps,
        "max_frames": options.max_frames,
        "max_video_seconds": options.max_video_seconds,
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "disable_cudnn": options.pipeline_options.disable_cudnn,
        "disable_sdp": options.pipeline_options.disable_sdp,
        "persistent_depth_services": options.pipeline_options.persistent_depth_services,
        "model_paths": {
            "pi3x": str(options.model_paths.pi3x),
            "moge3": str(options.model_paths.moge3),
            "vipe": str(options.model_paths.vipe),
        },
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


def _worker_main(
    local_worker_id: int,
    global_worker_id: int,
    gpu_id: int,
    tasks: list[Path],
    options: BatchOptions,
    checkpoint_path: Path,
    progress_queue: Any,
    pi3x_service: DepthServiceEndpoint | None,
    moge3_service: DepthServiceEndpoint | None,
) -> None:
    """Run one fixed task partition on one visible GPU and checkpoint every result."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_lease = TaskLease(
        checkpoint_path.parent
        / "worker_leases"
        / f"worker_{global_worker_id:03d}.lease",
        global_worker_id,
        options.lease_timeout_seconds,
    )
    if not worker_lease.acquire():
        for video in tasks:
            relative = _task_name(video)
            progress_queue.put(
                ("claimed_elsewhere", global_worker_id, relative, "")
            )
        progress_queue.put(("worker_finished", global_worker_id, "", ""))
        return
    pipeline_options = replace(
        options.pipeline_options,
        device="cuda:0",
        keep_work=False,
        pi3x_service=pi3x_service,
        moge3_service=moge3_service,
    )
    state: dict[str, Any] = {
        "format_version": 1,
        "worker_id": global_worker_id,
        "local_worker_id": local_worker_id,
        "global_worker_id": global_worker_id,
        "node_rank": options.node_rank,
        "num_nodes": options.num_nodes,
        "gpu_id": gpu_id,
        "assigned_tasks": [
            _task_name(path) for path in tasks
        ],
        "tasks": {},
    }
    if checkpoint_path.is_file():
        try:
            previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if previous.get("assigned_tasks") == state["assigned_tasks"]:
                state = previous
        except (OSError, json.JSONDecodeError):
            pass
    _atomic_checkpoint(checkpoint_path, state)
    for video in tasks:
        pending_output: Path | None = None
        relative = _task_name(video)
        job_key = hashlib.sha256(relative.encode()).hexdigest()[:16]
        job_root = checkpoint_path.parent / "stage_cache" / (
            f"worker_{global_worker_id:03d}_{job_key}"
        )
        if not options.overwrite and valid_existing_output(
            video,
            options.target_fps,
            options.max_frames,
            options.max_video_seconds,
        ):
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
            }
            _atomic_checkpoint(checkpoint_path, state)
            if not options.keep_stage_cache:
                shutil.rmtree(job_root, ignore_errors=True)
            progress_queue.put(("skipped", global_worker_id, relative, ""))
            continue
        lease = TaskLease(
            video_lease_path(video),
            global_worker_id,
            options.lease_timeout_seconds,
        )
        if not lease.acquire():
            state["tasks"][relative] = {
                "status": "claimed_elsewhere",
                "lease": str(lease.path),
            }
            _atomic_checkpoint(checkpoint_path, state)
            progress_queue.put(
                ("claimed_elsewhere", global_worker_id, relative, "")
            )
            continue
        try:
            validate_existing_output_owners([video])
        except ValueError as error:
            state["tasks"][relative] = {
                "status": "failed",
                "error": str(error),
            }
            _atomic_checkpoint(checkpoint_path, state)
            lease.release()
            progress_queue.put(("failed", global_worker_id, relative, str(error)))
            continue
        if not options.overwrite and valid_existing_output(
            video,
            options.target_fps,
            options.max_frames,
            options.max_video_seconds,
        ):
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
            }
            _atomic_checkpoint(checkpoint_path, state)
            lease.release()
            progress_queue.put(("skipped", global_worker_id, relative, ""))
            continue
        if options.overwrite and job_root.is_dir():
            shutil.rmtree(job_root)
        state["tasks"][relative] = {
            "status": "running",
            "stage_cache": str(job_root),
        }
        _atomic_checkpoint(checkpoint_path, state)
        try:
            normalized, source_fps, applied_seconds = prepare_video(
                video,
                job_root,
                options.target_fps,
                options.max_frames,
                options.max_video_seconds,
                options.ffmpeg_command,
            )
            result_dir = camera_artifact_dir(video)
            CameraCreatePipeline(options.model_paths, pipeline_options).run(
                normalized, result_dir, job_root / "pipeline"
            )
            pending_output = camera_json_path(video).with_name(
                f".{camera_json_path(video).name}.{lease.token}.pending"
            )
            lease.assert_owned()
            payload = export_camera_json_v2(
                result_dir,
                pending_output,
                video.name,
                source_fps,
                options.target_fps,
                options.max_frames,
                applied_seconds,
            )
            lease.assert_owned()
            pending_output.replace(camera_json_path(video))
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
                "frame_count": payload["frame_count"],
            }
            if not options.keep_stage_cache:
                shutil.rmtree(job_root, ignore_errors=True)
            event = ("completed", global_worker_id, relative, "")
        except Exception as error:  # noqa: BLE001 - isolate failure to one video
            if pending_output is not None:
                pending_output.unlink(missing_ok=True)
            detail = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            stage_state_path = job_root / "pipeline" / "stage_state.json"
            completed_stages: list[str] = []
            if stage_state_path.is_file():
                try:
                    completed_stages = json.loads(
                        stage_state_path.read_text(encoding="utf-8")
                    ).get("completed_stages", [])
                except (OSError, json.JSONDecodeError):
                    pass
            state["tasks"][relative] = {
                "status": "failed",
                "error": detail,
                "stage_cache": str(job_root),
                "completed_stages": completed_stages,
            }
            event = ("failed", global_worker_id, relative, str(error))
        finally:
            lease.release()
        _atomic_checkpoint(checkpoint_path, state)
        progress_queue.put(event)
    worker_lease.release()
    progress_queue.put(("worker_finished", global_worker_id, "", ""))


def run_batch(options: BatchOptions) -> dict[str, Any]:
    """Load, preassign, execute, resume, and summarize one manifest shard."""
    if not options.gpu_ids:
        raise ValueError("At least one GPU id is required")
    if options.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    if options.depth_services_per_gpu < 1:
        raise ValueError("depth_services_per_gpu must be positive")
    if (
        options.pipeline_options.persistent_depth_services
        and options.depth_services_per_gpu > options.workers_per_gpu
    ):
        raise ValueError("depth_services_per_gpu cannot exceed workers_per_gpu")
    if options.lease_timeout_seconds <= 0:
        raise ValueError("lease_timeout_seconds must be positive")
    if (
        options.target_fps <= 0
        or options.max_frames <= 0
        or options.max_video_seconds <= 0
    ):
        raise ValueError("target_fps, max_frames and max_video_seconds must be positive")
    options.model_paths.validate_depth_models()
    videos = load_video_manifest(options.input_manifest, options.extensions)
    validate_unique_camera_outputs(videos)
    validate_existing_output_owners(videos)
    local_worker_count = len(options.gpu_ids) * options.workers_per_gpu
    expected_launcher_processes = options.num_nodes * len(options.gpu_ids)
    if (
        options.launcher_num_processes is not None
        and options.launcher_num_processes != expected_launcher_processes
    ):
        raise ValueError(
            "--num-processes describes DLC GPU slots and must equal "
            f"num_nodes × GPUs_per_node = {expected_launcher_processes}; got "
            f"{options.launcher_num_processes}. --workers-per-gpu is applied separately."
        )
    layout = DistributedLayout(
        node_rank=options.node_rank,
        num_nodes=options.num_nodes,
        local_worker_count=local_worker_count,
    )
    layout.validate()
    if options.num_nodes > 1 and options.run_id is None:
        raise ValueError("--run-id is required when --num-nodes is greater than 1")
    run_token = (
        validate_run_id(options.run_id)
        if options.run_id is not None
        else _run_key(videos, options)
    )
    run_root = options.checkpoint_root / f"run_{run_token}"
    manifest_sha256 = ensure_shared_manifest(
        run_root, _manifest_payload(videos, options, layout)
    )
    node_lease = TaskLease(
        run_root / "node_leases" / f"node_{options.node_rank:03d}.lease",
        options.node_rank,
        options.lease_timeout_seconds,
    )
    if not node_lease.acquire():
        raise RuntimeError(
            f"Node rank {options.node_rank} is already active for run {run_token}. "
            "Every live machine must use a unique --node-rank."
        )
    atexit.register(node_lease.release)
    assignments = assign_node_tasks(videos, layout)
    videos_assigned = sum(len(tasks) for tasks in assignments)
    already_complete = sum(
        valid_existing_output(
            video,
            options.target_fps,
            options.max_frames,
            options.max_video_seconds,
        )
        for video in videos
    )
    service_gpu_ids = tuple(
        gpu_id
        for gpu_index, gpu_id in enumerate(options.gpu_ids)
        if options.pipeline_options.persistent_depth_services
        and (
            options.overwrite
            or any(
            not valid_existing_output(
                video,
                options.target_fps,
                options.max_frames,
                options.max_video_seconds,
            )
            for tasks in assignments[
                gpu_index
                * options.workers_per_gpu : (gpu_index + 1)
                * options.workers_per_gpu
            ]
            for video in tasks
            )
        )
    )
    summary = {
        "input_manifest": str(options.input_manifest),
        "videos_found": len(videos),
        "gpu_ids": list(options.gpu_ids),
        "workers_per_gpu": options.workers_per_gpu,
        "depth_services_per_gpu": options.depth_services_per_gpu,
        "node_rank": options.node_rank,
        "num_nodes": options.num_nodes,
        "local_workers": local_worker_count,
        "global_workers": layout.global_worker_count,
        "total_workers": layout.global_worker_count,
        "videos_assigned": videos_assigned,
        "run_id": run_token,
        "manifest_sha256": manifest_sha256,
        "launcher_num_processes": options.launcher_num_processes,
        "main_process_ip": options.main_process_ip,
        "main_process_port": options.main_process_port,
        "lease_timeout_seconds": options.lease_timeout_seconds,
        "target_fps": options.target_fps,
        "max_frames": options.max_frames,
        "max_video_seconds": options.max_video_seconds,
        "already_complete": already_complete,
        "checkpoint_root": str(run_root),
        "overwrite": options.overwrite,
        "keep_stage_cache": options.keep_stage_cache,
        "video_extensions": list(options.extensions),
        "ffmpeg_command": options.ffmpeg_command,
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "pi3x_python": str(options.pipeline_options.pi3x_python),
        "moge3_python": str(options.pipeline_options.moge3_python),
        "vipe_command": options.pipeline_options.vipe_command,
        "allow_vipe_downloads": options.pipeline_options.allow_vipe_downloads,
        "disable_cudnn": options.pipeline_options.disable_cudnn,
        "disable_sdp": options.pipeline_options.disable_sdp,
        "persistent_depth_services": options.pipeline_options.persistent_depth_services,
        "persistent_depth_service_gpu_ids": list(service_gpu_ids),
        "pi3x_checkpoint": str(options.model_paths.pi3x),
        "moge3_checkpoint": str(options.model_paths.moge3),
        "vipe_cache": str(options.model_paths.vipe),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not videos:
        result = {
            **summary,
            "completed": 0,
            "skipped": 0,
            "claimed_elsewhere": 0,
            "failed": 0,
            "worker_crashes": 0,
        }
        _write_batch_summary(run_root, options.node_rank, options.num_nodes, result)
        node_lease.release()
        atexit.unregister(node_lease.release)
        return result

    service_root: Path | None = None
    services: list[DepthServiceProcess] = []
    pi3x_services: dict[tuple[int, int], DepthServiceEndpoint] = {}
    moge3_services: dict[tuple[int, int], DepthServiceEndpoint] = {}

    def stop_depth_services() -> None:
        """Stop every persistent depth process owned by this controller."""
        for service in reversed(services):
            service.stop()
        if service_root is not None:
            shutil.rmtree(service_root, ignore_errors=True)

    if service_gpu_ids:
        service_root = Path(tempfile.mkdtemp(prefix="camera-create-depth-services-"))
        try:
            pi3x_group, pi3x_services = _start_depth_service_group(
                "Pi3X", service_gpu_ids, options, service_root
            )
            services.extend(pi3x_group)
            moge3_group, moge3_services = _start_depth_service_group(
                "MoGe-3", service_gpu_ids, options, service_root
            )
            services.extend(moge3_group)
        except Exception:
            stop_depth_services()
            raise
        atexit.register(stop_depth_services)

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[mp.Process] = []
    for local_worker_id, tasks in enumerate(assignments):
        global_worker_id = layout.global_worker_id(local_worker_id)
        gpu_id = options.gpu_ids[local_worker_id // options.workers_per_gpu]
        replica_id = depth_service_replica_id(
            local_worker_id,
            options.workers_per_gpu,
            options.depth_services_per_gpu,
        )
        checkpoint = run_root / f"worker_{global_worker_id:03d}.json"
        process = context.Process(
            target=_worker_main,
            args=(
                local_worker_id,
                global_worker_id,
                gpu_id,
                tasks,
                options,
                checkpoint,
                progress_queue,
                pi3x_services.get((gpu_id, replica_id)),
                moge3_services.get((gpu_id, replica_id)),
            ),
            name=f"camera-worker-{global_worker_id:03d}-gpu-{gpu_id}",
        )
        process.start()
        processes.append(process)

    counts = {
        "completed": 0,
        "skipped": 0,
        "claimed_elsewhere": 0,
        "failed": 0,
    }
    finished_workers = 0
    with tqdm(
        total=videos_assigned,
        desc=f"metric camera node {options.node_rank}/{options.num_nodes}",
        unit="video",
    ) as progress:
        while finished_workers < len(processes):
            try:
                status, worker_id, relative, detail = progress_queue.get(timeout=0.5)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if status == "worker_finished":
                finished_workers += 1
                continue
            counts[status] += 1
            progress.update(1)
            progress.set_postfix(counts, refresh=True)
            if status == "failed":
                tqdm.write(f"[worker {worker_id:03d}] FAILED {relative}: {detail}")

    crashed = 0
    for process in processes:
        process.join()
        if process.exitcode != 0:
            crashed += 1
    result = {**summary, **counts, "worker_crashes": crashed}
    _write_batch_summary(run_root, options.node_rank, options.num_nodes, result)
    if service_root is not None:
        stop_depth_services()
        atexit.unregister(stop_depth_services)
    node_lease.release()
    atexit.unregister(node_lease.release)
    return result
