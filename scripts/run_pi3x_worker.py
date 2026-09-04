#!/usr/bin/env python3
"""Decode one video and run Pi3X inside the dedicated Pi3X Python environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from camera_create.depth import infer_pi3x, load_pi3x
from camera_create.runtime import configure_torch_backends
from camera_create.video import read_video


def main() -> int:
    """Run Pi3X and atomically publish its relative-depth cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-inference-side", type=int, default=560)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--disable-sdp", action="store_true")
    args = parser.parse_args()
    configure_torch_backends(args.disable_cudnn, args.disable_sdp)
    video = read_video(args.input, args.max_inference_side)
    model = load_pi3x(args.checkpoint, args.device)
    depth = infer_pi3x(model, video.frames_rgb, args.device, args.chunk, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            depth=depth.astype(np.float32),
            frame_count=video.frame_count,
            original_width=video.original_width,
            original_height=video.original_height,
            inference_width=video.frames_rgb.shape[2],
            inference_height=video.frames_rgb.shape[1],
            fps=video.fps,
            model="pi3x",
            schema_version=1,
        )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
