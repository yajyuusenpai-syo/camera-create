"""Define the single end-to-end command-line interface for camera_create."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .batch import (
    DEFAULT_VIDEO_EXTENSIONS,
    BatchOptions,
    run_batch,
)
from .config import ModelPaths
from .pipeline import PipelineOptions
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


def _environment_bool(name: str) -> bool:
    """Interpret a conventional boolean environment variable."""
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Environment variable {name} must be a boolean: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments without importing heavyweight model dependencies."""
    parser = argparse.ArgumentParser(
        description="Estimate metric cameras for videos listed in a TXT/JSON shard"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="TXT or JSON file containing one shard of video paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Unsupported compatibility option; outputs stay beside each source video",
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
        help="Unsupported legacy single-video option",
    )
    parser.add_argument(
        "--stage-cache-dir",
        type=Path,
        help="Unsupported legacy single-video option; use --checkpoint-dir",
    )
    parser.add_argument(
        "--keep-stage-cache",
        action="store_true",
        help="Keep successful stage caches; failed runs always keep them for resume",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Unsupported legacy single-video option",
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
    parser.add_argument(
        "--no-persistent-depth-services",
        action="store_true",
        help="Reload Pi3X/MoGe-3 for every video instead of once per GPU",
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
        "--depth-services-per-gpu",
        type=int,
        default=4,
        help=(
            "Persistent Pi3X and MoGe-3 replicas per GPU; workers bind to replicas "
            "round-robin (default: 4)"
        ),
    )
    parser.add_argument(
        "--node-rank",
        "--machine-rank",
        "--machine_rank",
        dest="node_rank",
        type=int,
        default=0,
        help="Optional explicit node index; defaults to single-machine rank 0",
    )
    parser.add_argument(
        "--num-nodes",
        "--num-machines",
        "--num_machines",
        dest="num_nodes",
        type=int,
        default=1,
        help="Optional explicit machine count; defaults to one independent machine",
    )
    parser.add_argument(
        "--run-id",
        help="Checkpoint run namespace; generated from this shard when omitted",
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
        "--processing-height",
        type=int,
        default=480,
        help=(
            "Spatial height sent to the camera pipeline; aspect ratio is preserved "
            "(default: 480)"
        ),
    )
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated video extensions accepted in the input manifest",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint root; default: MANIFEST_DIR/.camera_create_ckpt/MANIFEST_STEM",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Batch mode: recompute valid existing cam_<video-stem>.json files",
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
        persistent_depth_services=not args.no_persistent_depth_services,
    )
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {input_path}")
    if input_path.suffix.lower() not in {".txt", ".json"}:
        raise ValueError("--input must be a .txt or .json video-path manifest")
    if args.output is not None:
        raise ValueError("--output is unsupported; outputs are written beside each video")
    if args.work_dir is not None or args.stage_cache_dir is not None or args.keep_work:
        raise ValueError(
            "--work-dir/--stage-cache-dir/--keep-work are unavailable in manifest mode"
        )
    extensions = tuple(
        item.strip().lower()
        for item in args.video_extensions.split(",")
        if item.strip()
    )
    checkpoint_root = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir
        else input_path.parent / ".camera_create_ckpt" / input_path.stem
    )
    report = run_batch(
        BatchOptions(
            input_manifest=input_path,
            checkpoint_root=checkpoint_root,
            model_paths=models,
            pipeline_options=options,
            gpu_ids=_parse_gpu_ids(args.gpu_ids),
            workers_per_gpu=args.workers_per_gpu,
            depth_services_per_gpu=args.depth_services_per_gpu,
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
            processing_height=args.processing_height,
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
