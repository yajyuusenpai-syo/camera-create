#!/usr/bin/env python3
"""Decode one video and run MoGe-3 inside the dedicated MoGe-3 Python environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing.connection import Listener
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


def infer_to_cache(
    args: argparse.Namespace,
    model,
    input_path: Path,
    output: Path,
    fov_x_deg: float | None,
) -> None:
    """Infer one video with an already-loaded MoGe-3 model and publish its cache."""
    video = read_video(input_path, args.max_inference_side)
    depths: list[np.ndarray] = []
    with torch.inference_mode():
        for frame in video.frames_rgb:
            image = torch.from_numpy(frame).permute(2, 0, 1).to(
                device=args.device, dtype=torch.float32
            ).div_(255.0)
            infer_options = {
                "refine_steps": args.refine_steps,
                "use_fp16": args.fp16,
            }
            if fov_x_deg is not None:
                infer_options["fov_x"] = fov_x_deg
            result = model.infer(image, **infer_options)
            depth = result["depth"].float()
            mask = result.get("mask")
            if mask is not None:
                depth = torch.where(mask.bool(), depth, torch.nan)
            if tuple(depth.shape[-2:]) != tuple(frame.shape[:2]):
                depth = torch.nn.functional.interpolate(
                    depth[None, None],
                    size=frame.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            depths.append(depth.cpu().numpy().astype(np.float32, copy=False))
    output_depth = np.stack(depths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
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
    temporary.replace(output)


def serve(args: argparse.Namespace, model) -> int:
    """Serve sequential MoGe-3 requests while retaining one GPU model instance."""
    listener = Listener(
        ("127.0.0.1", 0),
        family="AF_INET",
        backlog=128,
        authkey=bytes.fromhex(args.authkey),
    )
    host, port = listener.address
    ready = args.ready_file
    ready.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready.with_suffix(ready.suffix + ".tmp")
    temporary.write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    temporary.replace(ready)
    try:
        while True:
            connection = listener.accept()
            try:
                request = connection.recv()
                if request.get("command") == "shutdown":
                    connection.send({"ok": True})
                    return 0
                infer_to_cache(
                    args,
                    model,
                    Path(request["input"]),
                    Path(request["output"]),
                    request.get("fov_x_deg"),
                )
                connection.send({"ok": True})
            except Exception as error:  # noqa: BLE001 - isolate one request
                try:
                    connection.send(
                        {
                            "ok": False,
                            "error": "".join(
                                traceback.format_exception(
                                    type(error), error, error.__traceback__
                                )
                            ),
                        }
                    )
                except (EOFError, OSError):
                    pass
            finally:
                connection.close()
    finally:
        listener.close()
        ready.unlink(missing_ok=True)


def main() -> int:
    """Run per-frame metric inference and atomically publish the MoGe-3 cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-inference-side", type=int, default=560)
    parser.add_argument("--fov-x-deg", type=float)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--disable-sdp", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--authkey")
    args = parser.parse_args()
    configure_torch_backends(args.disable_cudnn, args.disable_sdp)
    model = load_moge3(args.checkpoint, args.device)
    if args.serve:
        if args.ready_file is None or not args.authkey:
            parser.error("--serve requires --ready-file and --authkey")
        return serve(args, model)
    if args.input is None or args.output is None:
        parser.error("--input and --output are required outside --serve mode")
    infer_to_cache(args, model, args.input, args.output, args.fov_x_deg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
