"""Define the single end-to-end command-line interface for camera_create."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import ModelPaths
from .pipeline import CameraCreatePipeline, PipelineOptions
from .worker_runner import default_environment_executable


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments without importing heavyweight model dependencies."""
    parser = argparse.ArgumentParser(
        description="Estimate metric camera intrinsics/extrinsics from a video"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
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
    parser.add_argument("--work-dir", type=Path, help="Explicit scratch directory")
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
    parser.add_argument("--moge3-refine-steps", type=int, default=3)
    parser.add_argument(
        "--moge3-no-fp16", action="store_true", help="Disable MoGe-3 mixed precision"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


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
    )
    report = CameraCreatePipeline(models, options).run(
        args.input, args.output, args.work_dir
    )
    print(json.dumps(report, indent=2))
    return 0
