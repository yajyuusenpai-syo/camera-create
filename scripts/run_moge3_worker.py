#!/usr/bin/env python3
"""Decode one video and run MoGe-3 inside the dedicated MoGe-3 Python environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from camera_create.runtime import configure_torch_backends
from camera_create.video import read_video


def load_moge3(checkpoint: Path, device: str):
    """Load a MoGe-3 checkpoint using the upstream v3 model class."""
    from moge.model.v3 import MoGeModel

    model_path = checkpoint / "model.pt" if (checkpoint / "model.pt").is_file() else checkpoint
    return MoGeModel.from_pretrained(str(model_path)).to(device).eval()


def main() -> int:
    """Run per-frame metric inference and atomically publish the MoGe-3 cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-inference-side", type=int, default=560)
    parser.add_argument("--fov-x-deg", type=float, default=60.0)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--disable-sdp", action="store_true")
    args = parser.parse_args()
    configure_torch_backends(args.disable_cudnn, args.disable_sdp)
    video = read_video(args.input, args.max_inference_side)
    model = load_moge3(args.checkpoint, args.device)
    depths: list[np.ndarray] = []
    with torch.inference_mode():
        for frame in video.frames_rgb:
            image = torch.from_numpy(frame).permute(2, 0, 1).to(
                device=args.device, dtype=torch.float32
            ).div_(255.0)
            result = model.infer(
                image,
                fov_x=args.fov_x_deg,
                refine_steps=args.refine_steps,
                use_fp16=args.fp16,
            )
            depth = result["depth"].float()
            if tuple(depth.shape[-2:]) != tuple(frame.shape[:2]):
                depth = torch.nn.functional.interpolate(
                    depth[None, None], size=frame.shape[:2], mode="bilinear", align_corners=False
                ).squeeze()
            depths.append(depth.cpu().numpy().astype(np.float32, copy=False))
    output_depth = np.stack(depths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            depth=output_depth,
            frame_count=video.frame_count,
            original_width=video.original_width,
            original_height=video.original_height,
            inference_width=video.frames_rgb.shape[2],
            inference_height=video.frames_rgb.shape[1],
            fps=video.fps,
            model="moge3",
            schema_version=1,
        )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
