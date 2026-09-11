#!/usr/bin/env python3
"""Serve repeated VIPE videos in one isolated process while resetting per-video state."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import torch


def configure_backends() -> None:
    """Apply the same global CUDA backend policy as one-shot VIPE."""
    if os.environ.get("CAMERA_CREATE_DISABLE_CUDNN") == "1":
        torch.backends.cudnn.enabled = False
    if os.environ.get("CAMERA_CREATE_DISABLE_SDP") == "1":
        cuda = torch.backends.cuda
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            function = getattr(cuda, name, None)
            if function is not None:
                function(False)
        enable_math = getattr(cuda, "enable_math_sdp", None)
        if enable_math is not None:
            enable_math(True)


def make_vipe_pipeline(output: Path):
    """Construct the patched cached-depth pipeline once for several videos."""
    from vipe import make_pipeline
    from vipe.config import parse_typed_config

    args = parse_typed_config(
        "default",
        hydra_args=[
            "pipeline=vipe_cached_depth",
            f"pipeline.output.path={output}",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_viz=false",
        ],
    )
    return make_pipeline(args.pipeline)


def infer(pipeline, video: Path, output: Path, cache_path: Path) -> None:
    """Reset the video-bound depth cache and run fresh SLAM state for one stream."""
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path.resolve())
    output.mkdir(parents=True, exist_ok=True)
    pipeline.out_path = output
    pipeline.out_cfg.path = str(output)
    # CachedDepthModel owns the NPZ of one video. Only this entry must be rebuilt;
    # GeoCalib and other heavyweight, stateless models remain resident.
    model_cache = getattr(pipeline, "model_cache", None)
    models = getattr(model_cache, "_models", None)
    if isinstance(models, dict):
        models.pop("depth/cached", None)
    stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(
        desc="Reading video stream"
    )
    pipeline.run(stream)


def main() -> int:
    """Listen on localhost and isolate every request failure from later videos."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--recycle-every", type=int, default=25)
    args = parser.parse_args()
    if args.recycle_every < 1:
        parser.error("--recycle-every must be positive")
    configure_backends()
    bootstrap_output = args.ready_file.parent / "bootstrap-output"
    pipeline = make_vipe_pipeline(bootstrap_output)
    listener = Listener(
        ("127.0.0.1", 0), family="AF_INET", backlog=128,
        authkey=bytes.fromhex(args.authkey),
    )
    host, port = listener.address
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
    temporary.write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    temporary.replace(args.ready_file)
    completed = 0
    try:
        while True:
            connection = listener.accept()
            try:
                request = connection.recv()
                if request.get("command") == "shutdown":
                    connection.send({"ok": True})
                    return 0
                infer(
                    pipeline,
                    Path(request["input"]),
                    Path(request["output"]),
                    Path(request["cache_path"]),
                )
                completed += 1
                connection.send({"ok": True})
                if completed % args.recycle_every == 0:
                    del pipeline
                    gc.collect()
                    torch.cuda.empty_cache()
                    pipeline = make_vipe_pipeline(bootstrap_output)
            except Exception as error:  # noqa: BLE001
                try:
                    connection.send(
                        {"ok": False, "error": "".join(traceback.format_exception(
                            type(error), error, error.__traceback__
                        ))}
                    )
                except (EOFError, OSError):
                    pass
                # A failed CUDA/SLAM request must not poison later videos. Rebuild
                # only after reporting the original traceback to the controller.
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
                pipeline = make_vipe_pipeline(bootstrap_output)
            finally:
                connection.close()
    finally:
        listener.close()
        args.ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
