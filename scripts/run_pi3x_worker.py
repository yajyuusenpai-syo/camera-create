#!/usr/bin/env python3
"""Decode one video and run Pi3X inside the dedicated Pi3X Python environment."""

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

from camera_create.depth import infer_pi3x, load_pi3x
from camera_create.runtime import configure_torch_backends
from camera_create.video import read_video


def infer_to_cache(args: argparse.Namespace, model, input_path: Path, output: Path) -> None:
    """Infer one video with an already-loaded Pi3X model and publish its cache."""
    video = read_video(input_path, args.max_inference_side)
    depth = infer_pi3x(model, video.frames_rgb, args.device, args.chunk, args.stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
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
    temporary.replace(output)


def serve(args: argparse.Namespace, model) -> int:
    """Serve sequential Pi3X requests while retaining one GPU model instance."""
    listener = Listener(("127.0.0.1", 0), family="AF_INET", backlog=128,
                        authkey=bytes.fromhex(args.authkey))
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
                    args, model, Path(request["input"]), Path(request["output"])
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
    """Run Pi3X and atomically publish its relative-depth cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-inference-side", type=int, default=560)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--disable-sdp", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--authkey")
    args = parser.parse_args()
    configure_torch_backends(args.disable_cudnn, args.disable_sdp)
    model = load_pi3x(args.checkpoint, args.device)
    if args.serve:
        if args.ready_file is None or not args.authkey:
            parser.error("--serve requires --ready-file and --authkey")
        return serve(args, model)
    if args.input is None or args.output is None:
        parser.error("--input and --output are required outside --serve mode")
    infer_to_cache(args, model, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
