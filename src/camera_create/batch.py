"""Recursively schedule metric-camera video jobs across isolated GPU worker processes."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from .artifacts import export_camera_json_v2
from .config import ModelPaths
from .pipeline import CameraCreatePipeline, PipelineOptions

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
    target_fps: float = 24.0
    max_video_seconds: float = 10.06
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS
    ffmpeg_command: str = "ffmpeg"
    overwrite: bool = False


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


def valid_existing_output(
    video: Path,
    target_fps: float | None = None,
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


def _prepare_video(
    source: Path,
    directory: Path,
    target_fps: float,
    max_seconds: float,
    ffmpeg_command: str,
) -> tuple[Path, float, float]:
    """Create a bounded constant-FPS processing copy and return its FPS metadata."""
    source_fps, _, _ = _probe_video(source)
    processed = directory / f"{source.stem}.normalized.mp4"
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
        str(processed),
    ]
    subprocess.run(command, check=True)
    _probe_video(processed)
    return processed, source_fps, max_seconds


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _run_key(videos: list[Path], options: BatchOptions) -> str:
    stable = {
        "videos": [path.relative_to(options.input_root).as_posix() for path in videos],
        "gpu_ids": options.gpu_ids,
        "workers_per_gpu": options.workers_per_gpu,
        "target_fps": options.target_fps,
        "max_video_seconds": options.max_video_seconds,
        "pi3x_chunk": options.pipeline_options.pi3x_chunk,
        "pi3x_stride": options.pipeline_options.pi3x_stride,
        "max_inference_side": options.pipeline_options.max_inference_side,
        "moge3_refine_steps": options.pipeline_options.moge3_refine_steps,
        "moge3_fp16": options.pipeline_options.moge3_fp16,
        "ema_momentum": options.pipeline_options.ema_momentum,
        "fov_x_deg": options.pipeline_options.fov_x_deg,
        "model_paths": {
            "pi3x": str(options.model_paths.pi3x),
            "moge3": str(options.model_paths.moge3),
            "vipe": str(options.model_paths.vipe),
        },
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


def _worker_main(
    worker_id: int,
    gpu_id: int,
    tasks: list[Path],
    options: BatchOptions,
    checkpoint_path: Path,
    progress_queue: Any,
) -> None:
    """Run one fixed task partition on one visible GPU and checkpoint every result."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pipeline_options = replace(
        options.pipeline_options, device="cuda:0", keep_work=False
    )
    state: dict[str, Any] = {
        "format_version": 1,
        "worker_id": worker_id,
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
        if not options.overwrite and valid_existing_output(
            video, options.target_fps, options.max_video_seconds
        ):
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
            }
            _atomic_checkpoint(checkpoint_path, state)
            progress_queue.put(("skipped", worker_id, relative, ""))
            continue
        state["tasks"][relative] = {"status": "running"}
        _atomic_checkpoint(checkpoint_path, state)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"camera-create-w{worker_id:03d}-"
            ) as temporary:
                job_root = Path(temporary)
                normalized, source_fps, applied_seconds = _prepare_video(
                    video,
                    job_root,
                    options.target_fps,
                    options.max_video_seconds,
                    options.ffmpeg_command,
                )
                result_dir = job_root / "result"
                CameraCreatePipeline(options.model_paths, pipeline_options).run(
                    normalized, result_dir
                )
                payload = export_camera_json_v2(
                    result_dir,
                    camera_json_path(video),
                    video.name,
                    source_fps,
                    options.target_fps,
                    applied_seconds,
                )
            state["tasks"][relative] = {
                "status": "completed",
                "output": str(camera_json_path(video)),
                "frame_count": payload["frame_count"],
            }
            event = ("completed", worker_id, relative, "")
        except Exception as error:  # noqa: BLE001 - isolate failure to one video
            detail = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            state["tasks"][relative] = {"status": "failed", "error": detail}
            event = ("failed", worker_id, relative, str(error))
        _atomic_checkpoint(checkpoint_path, state)
        progress_queue.put(event)
    progress_queue.put(("worker_finished", worker_id, "", ""))


def run_batch(options: BatchOptions) -> dict[str, Any]:
    """Scan, preassign, execute, resume, and summarize a multi-GPU video batch."""
    if not options.gpu_ids:
        raise ValueError("At least one GPU id is required")
    if options.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    if options.target_fps <= 0 or options.max_video_seconds <= 0:
        raise ValueError("target_fps and max_video_seconds must be positive")
    options.model_paths.validate_depth_models()
    videos = discover_videos(options.input_root, options.extensions)
    worker_count = len(options.gpu_ids) * options.workers_per_gpu
    assignments = assign_tasks(videos, worker_count)
    run_root = options.checkpoint_root / f"run_{_run_key(videos, options)}"
    run_root.mkdir(parents=True, exist_ok=True)
    already_complete = sum(
        valid_existing_output(video, options.target_fps, options.max_video_seconds)
        for video in videos
    )
    summary = {
        "input_root": str(options.input_root),
        "videos_found": len(videos),
        "gpu_ids": list(options.gpu_ids),
        "workers_per_gpu": options.workers_per_gpu,
        "total_workers": worker_count,
        "target_fps": options.target_fps,
        "max_video_seconds": options.max_video_seconds,
        "already_complete": already_complete,
        "checkpoint_root": str(run_root),
        "overwrite": options.overwrite,
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
            "failed": 0,
            "worker_crashes": 0,
        }
        (run_root / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[mp.Process] = []
    for worker_id, tasks in enumerate(assignments):
        gpu_id = options.gpu_ids[worker_id // options.workers_per_gpu]
        checkpoint = run_root / f"worker_{worker_id:03d}.json"
        if not checkpoint.exists():
            _atomic_checkpoint(
                checkpoint,
                {
                    "format_version": 1,
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "assigned_tasks": [
                        path.relative_to(options.input_root).as_posix()
                        for path in tasks
                    ],
                    "tasks": {},
                },
            )
        process = context.Process(
            target=_worker_main,
            args=(worker_id, gpu_id, tasks, options, checkpoint, progress_queue),
            name=f"camera-worker-{worker_id:03d}-gpu-{gpu_id}",
        )
        process.start()
        processes.append(process)

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    finished_workers = 0
    with tqdm(total=len(videos), desc="metric camera videos", unit="video") as progress:
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
    (run_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
