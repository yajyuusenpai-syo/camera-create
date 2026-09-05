"""Define the single end-to-end command-line interface for camera_create."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from .artifacts import export_camera_json_v2
from .batch import (
    DEFAULT_VIDEO_EXTENSIONS,
    BatchOptions,
    camera_artifact_dir,
    camera_json_path,
    prepare_video,
    run_batch,
)
from .config import ModelPaths
from .pipeline import CameraCreatePipeline, PipelineOptions
from .worker_runner import default_environment_executable


def _environment_int(name: str, default: int | None) -> int | None:
    """Read one optional integer launcher variable with an actionable error."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer: {value}") from error


def _environment_run_id() -> str | None:
    """Resolve an explicit or DLC-provided stable distributed run identifier."""
    for name in ("CAMERA_CREATE_RUN_ID", "DLC_JOB_ID", "PAI_JOB_ID"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _environment_bool(name: str) -> bool:
    """Interpret a conventional boolean environment variable."""
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Environment variable {name} must be a boolean: {value}")


def _distributed_environment_defaults() -> tuple[int, int]:
    """Resolve machine topology from DLC or an Accelerate/torchrun child process."""
    group_world_size = _environment_int("GROUP_WORLD_SIZE", None)
    group_rank = _environment_int("GROUP_RANK", None)
    if group_world_size is not None and group_rank is not None:
        return group_world_size, group_rank

    world_size = _environment_int("WORLD_SIZE", 1)
    rank = _environment_int("RANK", 0)
    local_world_size = _environment_int("LOCAL_WORLD_SIZE", None)
    if local_world_size is None or local_world_size <= 1:
        return world_size or 1, rank or 0
    if world_size is None or rank is None or world_size % local_world_size:
        raise ValueError(
            "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE for a multi-process "
            "DLC/Accelerate launch"
        )
    return world_size // local_world_size, rank // local_world_size


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments without importing heavyweight model dependencies."""
    environment_num_nodes, environment_node_rank = _distributed_environment_defaults()
    parser = argparse.ArgumentParser(
        description="Estimate metric camera intrinsics/extrinsics from a video"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input video file or directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Deprecated; outputs are always written beside each source video",
    )
    parser.add_argument("--ckpt-root", type=Path, help="Default: camera_create/ckpt")
    parser.add_argument("--pi3x-ckpt", type=Path, help="Override Pi3X checkpoint path")
    parser.add_argument(
        "--moge3-ckpt", type=Path, help="Override MoGe-3 checkpoint path"
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device, normally cuda or cuda:0"
    )
    parser.add_argument("--pi3x-chunk", type=int, default=16)
    parser.add_argument("--pi3x-stride", type=int, default=8)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--max-inference-side", type=int, default=560)
    parser.add_argument(
        "--fov-x-deg", type=float, help="Optional known horizontal field of view"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Legacy explicit scratch/resume directory for a single video",
    )
    parser.add_argument(
        "--stage-cache-dir",
        type=Path,
        help="Single-video stage resume directory; default: OUTPUT/.camera_create_ckpt",
    )
    parser.add_argument(
        "--keep-stage-cache",
        action="store_true",
        help="Keep successful stage caches; failed runs always keep them for resume",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Copy intermediate cache/VIPE data into output/work",
    )
    parser.add_argument(
        "--pi3x-python",
        type=Path,
        default=default_environment_executable("pi3x"),
        help="Python executable from the isolated Pi3X environment",
    )
    parser.add_argument(
        "--moge3-python",
        type=Path,
        default=default_environment_executable("moge3"),
        help="Python executable from the isolated MoGe-3 environment",
    )
    parser.add_argument(
        "--vipe-command",
        default=str(default_environment_executable("vipe", "vipe")),
        help="VIPE executable from the isolated VIPE environment",
    )
    parser.add_argument(
        "--allow-vipe-downloads",
        action="store_true",
        help="Allow VIPE to download missing runtime assets during inference",
    )
    parser.add_argument("--moge3-refine-steps", type=int, default=3)
    parser.add_argument(
        "--moge3-no-fp16", action="store_true", help="Disable MoGe-3 mixed precision"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        default=_environment_bool("CAMERA_CREATE_DISABLE_CUDNN"),
        help="Disable torch.backends.cudnn in Pi3X, MoGe-3 and VIPE",
    )
    parser.add_argument(
        "--disable-sdp",
        action="store_true",
        default=_environment_bool("CAMERA_CREATE_DISABLE_SDP"),
        help="Disable fused CUDA SDP backends globally while retaining math SDP",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help="Batch mode physical GPU ids, comma separated; for example 0,1,2,3",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=4,
        help="Batch pipelines per GPU; increase only after measuring VRAM",
    )
    parser.add_argument(
        "--node-rank",
        "--machine-rank",
        "--machine_rank",
        dest="node_rank",
        type=int,
        default=environment_node_rank,
        help="Batch node index; inferred from DLC or Accelerate launcher variables",
    )
    parser.add_argument(
        "--num-nodes",
        "--num-machines",
        "--num_machines",
        dest="num_nodes",
        type=int,
        default=environment_num_nodes,
        help="Total homogeneous machines; inferred from DLC or Accelerate variables",
    )
    parser.add_argument(
        "--run-id",
        default=_environment_run_id(),
        help="Shared immutable run namespace; required for multi-node batch mode",
    )
    parser.add_argument(
        "--num-processes",
        "--num_processes",
        dest="launcher_num_processes",
        type=int,
        help="Optional DLC/Accelerate total GPU process count; validated as nodes × GPUs",
    )
    parser.add_argument(
        "--main-process-ip",
        "--main_process_ip",
        dest="main_process_ip",
        default=os.environ.get("MASTER_ADDR"),
        help="DLC coordinator address; defaults to MASTER_ADDR and is recorded for audit",
    )
    parser.add_argument(
        "--main-process-port",
        "--main_process_port",
        dest="main_process_port",
        type=int,
        default=_environment_int("MASTER_PORT", None),
        help="DLC coordinator port; defaults to MASTER_PORT and is recorded for audit",
    )
    parser.add_argument(
        "--lease-timeout-seconds",
        type=float,
        default=900.0,
        help="Recover a per-video shared-filesystem lease after this heartbeat timeout",
    )
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=241,
        help="Maximum processed/output frames per video",
    )
    parser.add_argument("--max-video-seconds", type=float, default=10.06)
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated extensions recursively discovered in batch mode",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Batch checkpoint root; default: INPUT/.camera_create_ckpt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Batch mode: recompute valid existing cam_<video>.json files",
    )
    parser.add_argument("--ffmpeg-command", default="ffmpeg")
    return parser


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse and validate a unique comma-separated physical GPU list."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"Invalid --gpu-ids value: {value}") from error
    if (
        not values
        or any(value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("--gpu-ids must contain unique non-negative integers")
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end pipeline and print its machine-readable report."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    defaults = ModelPaths.defaults(args.ckpt_root)
    models = ModelPaths(
        pi3x=(args.pi3x_ckpt or defaults.pi3x).resolve(),
        moge3=(args.moge3_ckpt or defaults.moge3).resolve(),
        vipe=defaults.vipe,
    )
    options = PipelineOptions(
        device=args.device,
        pi3x_chunk=args.pi3x_chunk,
        pi3x_stride=args.pi3x_stride,
        ema_momentum=args.ema_momentum,
        max_inference_side=args.max_inference_side,
        fov_x_deg=args.fov_x_deg,
        keep_work=args.keep_work,
        pi3x_python=args.pi3x_python,
        moge3_python=args.moge3_python,
        vipe_command=args.vipe_command,
        moge3_refine_steps=args.moge3_refine_steps,
        moge3_fp16=not args.moge3_no_fp16,
        allow_vipe_downloads=args.allow_vipe_downloads,
        disable_cudnn=args.disable_cudnn,
        disable_sdp=args.disable_sdp,
    )
    input_path = args.input.resolve()
    if input_path.is_dir():
        local_rank = _environment_int("LOCAL_RANK", None)
        if local_rank not in (None, 0):
            print(
                json.dumps(
                    {
                        "status": "idle_launcher_process",
                        "local_rank": local_rank,
                        "reason": (
                            "LOCAL_RANK=0 owns this machine's camera_create workers"
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        if args.output is not None:
            raise ValueError("--output must be omitted in directory batch mode")
        if args.work_dir is not None or args.stage_cache_dir is not None or args.keep_work:
            raise ValueError(
                "--work-dir/--stage-cache-dir/--keep-work are single-video options"
            )
        extensions = tuple(
            item.strip().lower()
            for item in args.video_extensions.split(",")
            if item.strip()
        )
        checkpoint_root = (
            args.checkpoint_dir.resolve()
            if args.checkpoint_dir
            else input_path / ".camera_create_ckpt"
        )
        report = run_batch(
            BatchOptions(
                input_root=input_path,
                checkpoint_root=checkpoint_root,
                model_paths=models,
                pipeline_options=options,
                gpu_ids=_parse_gpu_ids(args.gpu_ids),
                workers_per_gpu=args.workers_per_gpu,
                node_rank=args.node_rank,
                num_nodes=args.num_nodes,
                run_id=args.run_id,
                launcher_num_processes=args.launcher_num_processes,
                main_process_ip=args.main_process_ip,
                main_process_port=args.main_process_port,
                lease_timeout_seconds=args.lease_timeout_seconds,
                target_fps=args.target_fps,
                max_frames=args.max_frames,
                max_video_seconds=args.max_video_seconds,
                extensions=extensions,
                ffmpeg_command=args.ffmpeg_command,
                overwrite=args.overwrite,
                keep_stage_cache=args.keep_stage_cache,
            )
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if any(
            report[name] for name in ("failed", "worker_crashes", "claimed_elsewhere")
        ) else 0
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if args.target_fps <= 0 or args.max_frames <= 0 or args.max_video_seconds <= 0:
        raise ValueError(
            "--target-fps, --max-frames and --max-video-seconds must be positive"
        )
    if args.output is not None:
        logging.getLogger(__name__).warning(
            "--output is deprecated and ignored; outputs are written beside %s",
            input_path,
        )
    if args.work_dir is not None and args.stage_cache_dir is not None:
        raise ValueError("--work-dir and --stage-cache-dir are mutually exclusive")
    output_dir = camera_artifact_dir(input_path)
    explicit_cache = args.work_dir is not None or args.stage_cache_dir is not None
    single_key = hashlib.sha256(str(input_path).encode()).hexdigest()[:16]
    stage_cache = (
        args.work_dir
        or args.stage_cache_dir
        or input_path.parent / ".camera_create_ckpt" / f"single_{single_key}"
    ).resolve()
    normalized, source_fps, applied_seconds = prepare_video(
        input_path,
        stage_cache,
        args.target_fps,
        args.max_frames,
        args.max_video_seconds,
        args.ffmpeg_command,
    )
    report = CameraCreatePipeline(models, options).run(
        normalized, output_dir, stage_cache / "pipeline"
    )
    payload = export_camera_json_v2(
        output_dir,
        camera_json_path(input_path),
        input_path.name,
        source_fps,
        args.target_fps,
        args.max_frames,
        applied_seconds,
    )
    if not explicit_cache and not args.keep_stage_cache:
        shutil.rmtree(stage_cache, ignore_errors=True)
    print(
        json.dumps(
            {
                "camera_json": str(camera_json_path(input_path)),
                "artifact_dir": str(output_dir),
                "frame_count": payload["frame_count"],
                "validation": report,
            },
            indent=2,
        )
    )
    return 0
