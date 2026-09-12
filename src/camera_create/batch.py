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

from .artifacts import CameraValidationError, export_camera_json_v2
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
from .video import inference_size
from .vipe_runner import (
    VipeServiceEndpoint,
    VipeServiceProcess,
    preflight_vipe_assets,
    preflight_vipe_integration,
    start_vipe_service,
)
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
    vipe_height: int = 720
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS
    ffmpeg_command: str = "ffmpeg"
    overwrite: bool = False
    keep_stage_cache: bool = False
    local_work_root: Path | None = None
    vipe_services_per_gpu: int = 1
    vipe_recycle_every: int = 25
    persistent_vipe: bool = True


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


def load_video_manifest(
    path: Path,
    extensions: tuple[str, ...],
    *,
    return_issues: bool = False,
) -> list[Path] | tuple[list[Path], list[dict[str, Any]]]:
    """Load a shard, warning and skipping individual unusable video entries.

    The manifest itself must exist and have a supported format.  A stale path in an
    otherwise valid shard is not a batch-level error: it is recorded as an issue so
    the remaining videos can still be processed.
    """
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
    issues: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for index, entry in enumerate(raw_entries, start=1):
        if isinstance(entry, dict):
            entry = entry.get("path", entry.get("video_path"))
        if not isinstance(entry, str) or not entry.strip():
            message = f"Invalid video path at manifest item {index}: {entry!r}"
            LOG.warning("Skipping manifest entry: %s", message)
            issues.append({"item": index, "path": None, "reason": "invalid_path", "message": message})
            continue
        candidate = Path(entry.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        candidate = candidate.resolve()
        if candidate in seen:
            message = f"Duplicate video path at manifest item {index}: {candidate}"
            LOG.warning("Skipping manifest entry: %s", message)
            issues.append({"item": index, "path": str(candidate), "reason": "duplicate", "message": message})
            continue
        if not candidate.is_file():
            message = f"Video listed at manifest item {index} does not exist: {candidate}"
            LOG.warning("Skipping manifest entry: %s", message)
            issues.append({"item": index, "path": str(candidate), "reason": "missing_file", "message": message})
            continue
        if candidate.suffix.lower() not in normalized_extensions:
            message = f"Unsupported video extension at manifest item {index}: {candidate}"
            LOG.warning("Skipping manifest entry: %s", message)
            issues.append({"item": index, "path": str(candidate), "reason": "unsupported_extension", "message": message})
            continue
        seen.add(candidate)
        videos.append(candidate)
    if return_issues:
        return videos, issues
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


def _start_vipe_service_group(
    gpu_ids: tuple[int, ...], options: BatchOptions, service_root: Path
) -> tuple[list[VipeServiceProcess], dict[tuple[int, int], VipeServiceEndpoint]]:
    """Start reusable VIPE pipelines in parallel waves across physical GPUs."""
    LOG.info(
        "Loading %d persistent VIPE services per GPU on GPUs %s",
        options.vipe_services_per_gpu,
        ",".join(str(gpu_id) for gpu_id in gpu_ids),
    )
    completed: dict[tuple[int, int], VipeServiceProcess] = {}
    failures: list[BaseException] = []
    for replica_id in range(options.vipe_services_per_gpu):
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(
                    start_vipe_service,
                    options.pipeline_options.vipe_command,
                    options.model_paths.vipe,
                    gpu_id,
                    service_root / f"vipe_gpu_{gpu_id}_replica_{replica_id}.json",
                    allow_downloads=options.pipeline_options.allow_vipe_downloads,
                    disable_cudnn=options.pipeline_options.disable_cudnn,
                    disable_sdp=options.pipeline_options.disable_sdp,
                    recycle_every=options.vipe_recycle_every,
                    assets_preflight_done=True,
                ): gpu_id
                for gpu_id in gpu_ids
            }
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    completed[(gpu_id, replica_id)] = future.result()
                    LOG.info(
                        "Persistent VIPE service replica %d/%d ready on GPU %d",
                        replica_id + 1,
                        options.vipe_services_per_gpu,
                        gpu_id,
                    )
                except BaseException as error:  # noqa: BLE001
                    failures.append(error)
        if failures:
            break
    if failures:
        for service in completed.values():
            service.stop()
        raise RuntimeError("Failed to load persistent VIPE service group") from failures[0]
    keys = [
        (gpu_id, replica_id)
        for replica_id in range(options.vipe_services_per_gpu)
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
    vipe_height: int | None = None,
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
        if vipe_height is not None:
            resolution = data.get("intrinsics_inference_resolution", {})
            valid &= int(resolution.get("height", -1)) == vipe_height
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
        "vipe_services_per_gpu": options.vipe_services_per_gpu,
        "vipe_recycle_every": options.vipe_recycle_every,
        "persistent_vipe": options.persistent_vipe,
        "target_fps": options.target_fps,
        "max_frames": options.max_frames,
        "max_video_seconds": options.max_video_seconds,
        "vipe_height": options.vipe_height,
        "extensions": list(options.extensions),
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "moge3_fov_policy": "model_estimated_when_unspecified_v2",
        "disable_cudnn": options.pipeline_options.disable_cudnn,
        "disable_sdp": options.pipeline_options.disable_sdp,
        "persistent_depth_services": options.pipeline_options.persistent_depth_services,
        "model_paths": {
            "pi3x": str(options.model_paths.pi3x),
            "moge3": str(options.model_paths.moge3),
            "vipe": str(options.model_paths.vipe),
        },
    }


def _probe_video(path: Path) -> tuple[float, int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Video reports invalid FPS or resolution: {path}")
    return fps, frames, frames / fps if frames > 0 else 0.0, width, height


def prepare_video(
    source: Path,
    directory: Path,
    target_fps: float,
    max_frames: int,
    max_seconds: float,
    vipe_height: int,
    ffmpeg_command: str,
    max_inference_side: int = 560,
) -> tuple[Path, Path, float, float, tuple[int, int], tuple[int, int]]:
    """Decode once and create aligned depth-sized and fixed-height VIPE videos."""
    directory.mkdir(parents=True, exist_ok=True)
    source_fps, _, _, source_width, source_height = _probe_video(source)
    depth_width, depth_height = inference_size(
        source_width, source_height, max_inference_side
    )
    processed = directory / f"{source.stem}.depth_{max_inference_side}.mp4"
    vipe_video = directory / f"{source.stem}.vipe_{vipe_height}p.mp4"
    marker = directory / "normalized.json"
    expected = {
        "source": video_identity(source),
        "target_fps": target_fps,
        "max_frames": max_frames,
        "max_seconds": max_seconds,
        "vipe_height": vipe_height,
        "max_inference_side": max_inference_side,
        "preparation_schema": 2,
    }
    if processed.is_file() and marker.is_file():
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
            processed_fps, processed_frames, _, cached_depth_width, cached_depth_height = _probe_video(
                processed
            )
            cached_vipe = (
                processed
                if cached_depth_height == vipe_height
                else vipe_video
            )
            if (
                state == expected
                and abs(processed_fps - target_fps) < 1e-3
                and (cached_depth_width, cached_depth_height)
                == (depth_width, depth_height)
                and cached_vipe.is_file()
            ):
                _, vipe_frames, _, vipe_width, cached_vipe_height = _probe_video(
                    cached_vipe
                )
                if vipe_frames != processed_frames:
                    raise ValueError("Cached depth and VIPE videos have different frames")
                return (
                    processed,
                    cached_vipe,
                    source_fps,
                    max_seconds,
                    (source_width, source_height),
                    (vipe_width, cached_vipe_height),
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    executable = shutil.which(ffmpeg_command)
    if executable is None:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_command}")
    vipe_width = max(2, round(source_width * vipe_height / source_height / 2) * 2)
    if (depth_width, depth_height) == (vipe_width, vipe_height):
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-t", str(max_seconds),
            "-vf", f"fps={target_fps},scale={depth_width}:{depth_height}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-frames:v", str(max_frames), str(processed),
        ]
        subprocess.run(command, check=True)
        vipe_input = processed
    else:
        # One decoder and one filter graph feed both encoders. This avoids reading
        # the source twice and avoids writing a full-resolution intermediate.
        filter_graph = (
            f"[0:v]trim=duration={max_seconds},setpts=PTS-STARTPTS,"
            f"fps={target_fps},split=2[depth_src][vipe_src];"
            f"[depth_src]scale={depth_width}:{depth_height}[depth];"
            f"[vipe_src]scale={vipe_width}:{vipe_height}[vipe]"
        )
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-filter_complex", filter_graph,
            "-map", "[depth]", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(max_frames),
            str(processed),
            "-map", "[vipe]", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(max_frames),
            str(vipe_video),
        ]
        subprocess.run(command, check=True)
        vipe_input = vipe_video
    _, processed_frames, _, _, _ = _probe_video(processed)
    _, vipe_frames, _, vipe_width, actual_vipe_height = _probe_video(vipe_input)
    if vipe_frames != processed_frames:
        raise RuntimeError(
            "Depth and VIPE processing videos have different frame counts: "
            f"{processed_frames} != {vipe_frames}"
        )
    _atomic_checkpoint(marker, expected)
    return (
        processed,
        vipe_input,
        source_fps,
        max_seconds,
        (source_width, source_height),
        (vipe_width, actual_vipe_height),
    )


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


def _write_failure_report(
    input_manifest: Path,
    run_root: Path,
    run_id: str,
    node_rank: int,
    num_nodes: int,
    worker_crashes: int,
) -> tuple[Path, Path]:
    """Collect full per-video tracebacks into a report beside the input shard."""
    failures: list[dict[str, Any]] = []
    for checkpoint in sorted(run_root.glob("worker_*.json")):
        try:
            worker = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for video_path, state in worker.get("tasks", {}).items():
            if not isinstance(state, dict) or state.get("status") not in {
                "failed",
                "rejected",
            }:
                continue
            completed = list(state.get("completed_stages", []))
            if state.get("failed_stage"):
                likely_stage = str(state["failed_stage"])
            elif not completed:
                likely_stage = "video_preparation_or_pi3x"
            elif completed[-1] == "pi3x":
                likely_stage = "moge3"
            elif completed[-1] == "moge3":
                likely_stage = "metric_depth_fusion"
            elif completed[-1] == "metric_depth":
                likely_stage = "vipe"
            else:
                likely_stage = "camera_export"
            failures.append(
                {
                    "video_path": video_path,
                    "outcome": state.get("status"),
                    "failure_kind": state.get("failure_kind", "runtime_error"),
                    "worker_id": worker.get("global_worker_id", worker.get("worker_id")),
                    "gpu_id": worker.get("gpu_id"),
                    "likely_failed_stage": likely_stage,
                    "completed_stages": completed,
                    "stage_cache": state.get("stage_cache"),
                    "error": state.get("error", "<missing traceback>"),
                    "validation": state.get("validation"),
                    "worker_checkpoint": str(checkpoint),
                }
            )
    failures.sort(key=lambda item: item["video_path"])
    rejected_count = sum(item["outcome"] == "rejected" for item in failures)
    failed_count = sum(item["outcome"] == "failed" for item in failures)
    node_suffix = "" if num_nodes == 1 else f".node_{node_rank:03d}"
    report_path = input_manifest.with_name(
        f"{input_manifest.stem}.camera_create_failures{node_suffix}.json"
    )
    failed_manifest_path = input_manifest.with_name(
        f"{input_manifest.stem}.camera_create_failed{node_suffix}.txt"
    )
    _atomic_checkpoint(
        report_path,
        {
            "format_version": 1,
            "input_manifest": str(input_manifest),
            "run_id": run_id,
            "node_rank": node_rank,
            "num_nodes": num_nodes,
            "issue_count": len(failures),
            "failed_count": failed_count,
            "rejected_count": rejected_count,
            "worker_crashes": worker_crashes,
            "failures": failures,
        },
    )
    temporary_manifest = failed_manifest_path.with_suffix(
        failed_manifest_path.suffix + ".tmp"
    )
    failed_lines = "".join(f"{item['video_path']}\n" for item in failures)
    temporary_manifest.write_text(failed_lines, encoding="utf-8")
    temporary_manifest.replace(failed_manifest_path)
    return report_path, failed_manifest_path


def _write_manifest_skip_report(
    input_manifest: Path, issues: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Write stale or malformed manifest entries beside their source shard."""
    report_path = input_manifest.with_name(
        f"{input_manifest.stem}.camera_create_manifest_skipped.json"
    )
    skipped_manifest_path = input_manifest.with_name(
        f"{input_manifest.stem}.camera_create_manifest_skipped.txt"
    )
    _atomic_checkpoint(
        report_path,
        {
            "format_version": 1,
            "input_manifest": str(input_manifest),
            "issue_count": len(issues),
            "issues": issues,
        },
    )
    lines = "".join(f"{item['path']}\n" for item in issues if item.get("path"))
    temporary = skipped_manifest_path.with_suffix(skipped_manifest_path.suffix + ".tmp")
    temporary.write_text(lines, encoding="utf-8")
    temporary.replace(skipped_manifest_path)
    return report_path, skipped_manifest_path


def _timing_stats(values: list[float]) -> dict[str, float | int | None]:
    """Summarize measured pure-inference durations without external dependencies."""
    if not values:
        return {"count": 0, "total_seconds": 0.0, "mean_seconds": None,
                "p50_seconds": None, "p90_seconds": None}
    ordered = sorted(values)
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "count": len(ordered),
        "total_seconds": sum(ordered),
        "mean_seconds": sum(ordered) / len(ordered),
        "p50_seconds": percentile(0.50),
        "p90_seconds": percentile(0.90),
    }


def _write_runtime_report(
    input_manifest: Path, run_root: Path, run_id: str, node_rank: int, num_nodes: int
) -> Path:
    """Collect per-model inference-only timings into one report beside the shard."""
    videos: list[dict[str, Any]] = []
    pure: dict[str, list[float]] = {"pi3x": [], "moge3": [], "vipe": []}
    vipe_cold_requests = 0
    for checkpoint in sorted(run_root.glob("worker_*.json")):
        try:
            worker = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for video_path, state in worker.get("tasks", {}).items():
            if not isinstance(state, dict) or not isinstance(state.get("model_timings"), dict):
                continue
            timings = state["model_timings"]
            videos.append({"video_path": video_path, "worker_id": worker.get("global_worker_id"),
                           "gpu_id": worker.get("gpu_id"), "models": timings})
            for model, model_values in pure.items():
                model_timing = timings.get(model, {})
                if not isinstance(model_timing, dict):
                    continue
                value = model_timing.get("pure_inference_seconds")
                if isinstance(value, (int, float)):
                    model_values.append(float(value))
            vipe_timing = timings.get("vipe", {})
            if isinstance(vipe_timing, dict) and vipe_timing.get("model_cache_cold") is True:
                vipe_cold_requests += 1
    videos.sort(key=lambda item: item["video_path"])
    suffix = "" if num_nodes == 1 else f".node_{node_rank:03d}"
    path = input_manifest.with_name(
        f"{input_manifest.stem}.camera_create_runtime{suffix}.json"
    )
    _atomic_checkpoint(
        path,
        {
            "format_version": 1,
            "input_manifest": str(input_manifest),
            "run_id": run_id,
            "node_rank": node_rank,
            "num_nodes": num_nodes,
            "timing_definition": (
                "Pi3X/MoGe-3 time covers model forward only. VIPE pure time excludes "
                "cold requests that load/recycle its model cache."
            ),
            "models": {name: _timing_stats(values) for name, values in pure.items()},
            "vipe_cold_requests_excluded": vipe_cold_requests,
            "videos_with_timings": len(videos),
            "videos": videos,
        },
    )
    return path


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
        "vipe_height": options.vipe_height,
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "moge3_fov_policy": "model_estimated_when_unspecified_v2",
        "disable_cudnn": options.pipeline_options.disable_cudnn,
        "disable_sdp": options.pipeline_options.disable_sdp,
        "persistent_depth_services": options.pipeline_options.persistent_depth_services,
        "persistent_vipe": options.persistent_vipe,
        "vipe_services_per_gpu": options.vipe_services_per_gpu,
        "vipe_recycle_every": options.vipe_recycle_every,
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
    vipe_service: VipeServiceEndpoint | None,
    local_run_root: Path,
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
        vipe_service=vipe_service,
        preflight_done=True,
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
        pipeline_report: dict[str, Any] | None = None
        relative = _task_name(video)
        job_key = hashlib.sha256(relative.encode()).hexdigest()[:16]
        job_root = local_run_root / "stage_cache" / (
            f"worker_{global_worker_id:03d}_{job_key}"
        )
        if not options.overwrite and valid_existing_output(
            video,
            options.target_fps,
            options.max_frames,
            options.max_video_seconds,
            options.vipe_height,
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
            options.vipe_height,
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
            (
                normalized,
                vipe_video,
                source_fps,
                applied_seconds,
                source_resolution,
                intrinsics_inference_resolution,
            ) = prepare_video(
                video,
                job_root,
                options.target_fps,
                options.max_frames,
                options.max_video_seconds,
                options.vipe_height,
                options.ffmpeg_command,
                options.pipeline_options.max_inference_side,
            )
            result_dir = camera_artifact_dir(video)
            pipeline_report = CameraCreatePipeline(options.model_paths, pipeline_options).run(
                normalized,
                result_dir,
                job_root / "pipeline",
                vipe_video=vipe_video,
                vipe_resolution=intrinsics_inference_resolution,
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
                source_resolution,
                intrinsics_inference_resolution,
            )
            lease.assert_owned()
            pending_output.replace(camera_json_path(video))
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
                "frame_count": payload["frame_count"],
                "model_timings": pipeline_report.get("model_timings", {}),
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
            rejected = isinstance(error, CameraValidationError)
            state["tasks"][relative] = {
                "status": "rejected" if rejected else "failed",
                "failure_kind": "data_validation" if rejected else "runtime_error",
                "failed_stage": "camera_validation" if rejected else None,
                "error": detail,
                "validation": error.report if rejected else None,
                "stage_cache": str(job_root),
                "completed_stages": completed_stages,
                "model_timings": (
                    pipeline_report.get("model_timings", {}) if pipeline_report else {}
                ),
            }
            event = (
                "rejected" if rejected else "failed",
                global_worker_id,
                relative,
                str(error),
            )
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
    if options.vipe_services_per_gpu < 1:
        raise ValueError("vipe_services_per_gpu must be positive")
    if options.vipe_recycle_every < 1:
        raise ValueError("vipe_recycle_every must be positive")
    if (
        options.pipeline_options.persistent_depth_services
        and options.depth_services_per_gpu > options.workers_per_gpu
    ):
        raise ValueError("depth_services_per_gpu cannot exceed workers_per_gpu")
    if options.persistent_vipe and options.vipe_services_per_gpu > options.workers_per_gpu:
        raise ValueError("vipe_services_per_gpu cannot exceed workers_per_gpu")
    if options.lease_timeout_seconds <= 0:
        raise ValueError("lease_timeout_seconds must be positive")
    if (
        options.target_fps <= 0
        or options.max_frames <= 0
        or options.max_video_seconds <= 0
    ):
        raise ValueError("target_fps, max_frames and max_video_seconds must be positive")
    if options.vipe_height < 2 or options.vipe_height % 2:
        raise ValueError("vipe_height must be a positive even integer")
    options.model_paths.validate_depth_models()
    manifest_result = load_video_manifest(
        options.input_manifest, options.extensions, return_issues=True
    )
    assert isinstance(manifest_result, tuple)
    videos, manifest_issues = manifest_result
    manifest_skip_report, manifest_skipped_list = _write_manifest_skip_report(
        options.input_manifest, manifest_issues
    )
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
    local_base = (
        options.local_work_root.resolve()
        if options.local_work_root is not None
        else Path(tempfile.gettempdir()).resolve() / "camera-create"
    )
    local_namespace = hashlib.sha256(str(run_root.resolve()).encode()).hexdigest()[:12]
    local_run_root = local_base / f"run_{run_token}_{local_namespace}"
    local_run_root.mkdir(parents=True, exist_ok=True)
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
            options.vipe_height,
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
                options.vipe_height,
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
    vipe_gpu_ids = tuple(
        gpu_id
        for gpu_index, gpu_id in enumerate(options.gpu_ids)
        if options.persistent_vipe
        and (
            options.overwrite
            or any(
                not valid_existing_output(
                    video,
                    options.target_fps,
                    options.max_frames,
                    options.max_video_seconds,
                    options.vipe_height,
                )
                for tasks in assignments[
                    gpu_index * options.workers_per_gpu : (gpu_index + 1)
                    * options.workers_per_gpu
                ]
                for video in tasks
            )
        )
    )
    summary = {
        "input_manifest": str(options.input_manifest),
        "videos_found": len(videos),
        "manifest_entries_skipped": len(manifest_issues),
        "manifest_skip_report": str(manifest_skip_report),
        "manifest_skipped_list": str(manifest_skipped_list),
        "gpu_ids": list(options.gpu_ids),
        "workers_per_gpu": options.workers_per_gpu,
        "depth_services_per_gpu": options.depth_services_per_gpu,
        "vipe_services_per_gpu": options.vipe_services_per_gpu,
        "vipe_recycle_every": options.vipe_recycle_every,
        "persistent_vipe": options.persistent_vipe,
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
        "vipe_height": options.vipe_height,
        "already_complete": already_complete,
        "checkpoint_root": str(run_root),
        "local_work_root": str(local_run_root),
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
        "persistent_vipe_service_gpu_ids": list(vipe_gpu_ids),
        "pi3x_checkpoint": str(options.model_paths.pi3x),
        "moge3_checkpoint": str(options.model_paths.moge3),
        "vipe_cache": str(options.model_paths.vipe),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not videos:
        failure_report, failed_manifest = _write_failure_report(
            options.input_manifest,
            run_root,
            run_token,
            options.node_rank,
            options.num_nodes,
            0,
        )
        runtime_report = _write_runtime_report(
            options.input_manifest,
            run_root,
            run_token,
            options.node_rank,
            options.num_nodes,
        )
        result = {
            **summary,
            "completed": 0,
            "skipped": 0,
            "rejected": 0,
            "claimed_elsewhere": 0,
            "failed": 0,
            "worker_crashes": 0,
            "failure_report": str(failure_report),
            "failed_manifest": str(failed_manifest),
            "runtime_report": str(runtime_report),
        }
        _write_batch_summary(run_root, options.node_rank, options.num_nodes, result)
        node_lease.release()
        atexit.unregister(node_lease.release)
        return result

    preflight_vipe_assets(
        options.model_paths.vipe, options.pipeline_options.allow_vipe_downloads
    )
    preflight_vipe_integration(options.pipeline_options.vipe_command)
    LOG.info("Machine-level VIPE asset/integration preflight passed")
    LOG.info("Using local intermediate-data root: %s", local_run_root)

    service_root: Path | None = None
    services: list[DepthServiceProcess] = []
    vipe_processes: list[VipeServiceProcess] = []
    pi3x_services: dict[tuple[int, int], DepthServiceEndpoint] = {}
    moge3_services: dict[tuple[int, int], DepthServiceEndpoint] = {}
    vipe_services: dict[tuple[int, int], VipeServiceEndpoint] = {}

    def stop_depth_services() -> None:
        """Stop every persistent depth process owned by this controller."""
        for service in reversed(services):
            service.stop()
        if service_root is not None:
            shutil.rmtree(service_root, ignore_errors=True)

    def stop_vipe_services() -> None:
        """Stop every persistent VIPE process owned by this controller."""
        for service in reversed(vipe_processes):
            service.stop()

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

    if vipe_gpu_ids and options.persistent_vipe:
        if service_root is None:
            service_root = Path(tempfile.mkdtemp(prefix="camera-create-services-"))
        try:
            vipe_processes, vipe_services = _start_vipe_service_group(
                vipe_gpu_ids, options, service_root
            )
        except Exception:
            stop_depth_services()
            raise
        atexit.register(stop_vipe_services)

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
        vipe_replica_id = depth_service_replica_id(
            local_worker_id,
            options.workers_per_gpu,
            options.vipe_services_per_gpu,
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
                vipe_services.get((gpu_id, vipe_replica_id)),
                local_run_root,
            ),
            name=f"camera-worker-{global_worker_id:03d}-gpu-{gpu_id}",
        )
        process.start()
        processes.append(process)

    counts = {
        "completed": 0,
        "skipped": 0,
        "rejected": 0,
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
    failure_report, failed_manifest = _write_failure_report(
        options.input_manifest,
        run_root,
        run_token,
        options.node_rank,
        options.num_nodes,
        crashed,
    )
    runtime_report = _write_runtime_report(
        options.input_manifest, run_root, run_token, options.node_rank, options.num_nodes
    )
    result = {
        **summary,
        **counts,
        "worker_crashes": crashed,
        "failure_report": str(failure_report),
        "failed_manifest": str(failed_manifest),
        "runtime_report": str(runtime_report),
    }
    _write_batch_summary(run_root, options.node_rank, options.num_nodes, result)
    if service_root is not None:
        if vipe_processes:
            stop_vipe_services()
            atexit.unregister(stop_vipe_services)
        stop_depth_services()
        atexit.unregister(stop_depth_services)
    node_lease.release()
    atexit.unregister(node_lease.release)
    return result
