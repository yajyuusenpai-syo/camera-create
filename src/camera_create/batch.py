"""Recursively schedule metric-camera video jobs across isolated GPU worker processes."""

from __future__ import annotations

import atexit
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import traceback
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


@dataclass(frozen=True)
class BatchOptions:
    """Serializable settings shared by all statically assigned batch workers."""

    input_root: Path
    checkpoint_root: Path
    model_paths: ModelPaths
    pipeline_options: PipelineOptions
    gpu_ids: tuple[int, ...]
    workers_per_gpu: int = 1
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


def camera_json_path(video: Path) -> Path:
    """Place one camera JSON beside its source while preserving the full filename."""
    return video.parent / f"cam_{video.name}.json"


def camera_artifact_dir(video: Path) -> Path:
    """Place NPY/report artifacts beside the source without stem collisions."""
    return video.parent / f"{video.name}.camera"


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


def _manifest_payload(
    videos: list[Path], options: BatchOptions, layout: DistributedLayout
) -> dict[str, Any]:
    """Build the immutable task/config contract every distributed node must share."""
    return {
        "format_version": 1,
        "videos": [
            {
                "relative_path": path.relative_to(options.input_root).as_posix(),
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
                "relative_path": path.relative_to(options.input_root).as_posix(),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in videos
        ],
        "gpu_ids": options.gpu_ids,
        "workers_per_gpu": options.workers_per_gpu,
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
            relative = video.relative_to(options.input_root).as_posix()
            progress_queue.put(
                ("claimed_elsewhere", global_worker_id, relative, "")
            )
        progress_queue.put(("worker_finished", global_worker_id, "", ""))
        return
    pipeline_options = replace(
        options.pipeline_options, device="cuda:0", keep_work=False
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
            path.relative_to(options.input_root).as_posix() for path in tasks
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
        relative = video.relative_to(options.input_root).as_posix()
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
            checkpoint_path.parent / "leases" / f"{job_key}.lease",
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
            payload = export_camera_json_v2(
                result_dir,
                camera_json_path(video),
                video.name,
                source_fps,
                options.target_fps,
                options.max_frames,
                applied_seconds,
            )
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
                "frame_count": payload["frame_count"],
            }
            if not options.keep_stage_cache:
                shutil.rmtree(job_root, ignore_errors=True)
            event = ("completed", global_worker_id, relative, "")
        except Exception as error:  # noqa: BLE001 - isolate failure to one video
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
    """Scan, preassign, execute, resume, and summarize a multi-GPU video batch."""
    if not options.gpu_ids:
        raise ValueError("At least one GPU id is required")
    if options.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    if options.lease_timeout_seconds <= 0:
        raise ValueError("lease_timeout_seconds must be positive")
    if (
        options.target_fps <= 0
        or options.max_frames <= 0
        or options.max_video_seconds <= 0
    ):
        raise ValueError("target_fps, max_frames and max_video_seconds must be positive")
    options.model_paths.validate_depth_models()
    videos = discover_videos(options.input_root, options.extensions)
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
    summary = {
        "input_root": str(options.input_root),
        "videos_found": len(videos),
        "gpu_ids": list(options.gpu_ids),
        "workers_per_gpu": options.workers_per_gpu,
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

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[mp.Process] = []
    for local_worker_id, tasks in enumerate(assignments):
        global_worker_id = layout.global_worker_id(local_worker_id)
        gpu_id = options.gpu_ids[local_worker_id // options.workers_per_gpu]
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
    node_lease.release()
    atexit.unregister(node_lease.release)
    return result
