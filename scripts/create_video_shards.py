#!/usr/bin/env python3
"""Recursively discover videos and write balanced TXT/JSON path shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_create.batch import (
    DEFAULT_VIDEO_EXTENSIONS,
    discover_videos,
    validate_unique_camera_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Define shard-generation arguments without loading any model package."""
    parser = argparse.ArgumentParser(
        description="Split recursively discovered video paths into balanced shards"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--prefix", default="clip")
    parser.add_argument("--format", choices=("txt", "json"), default="txt")
    parser.add_argument(
        "--video-extensions", default=",".join(DEFAULT_VIDEO_EXTENSIONS)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create deterministic round-robin shards whose sizes differ by at most one."""
    args = build_parser().parse_args(argv)
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    source = args.input_dir.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Input video directory does not exist: {source}")
    extensions = tuple(
        item.strip().lower()
        for item in args.video_extensions.split(",")
        if item.strip()
    )
    videos = discover_videos(source, extensions)
    validate_unique_camera_outputs(videos)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    widths = max(1, len(str(args.shards)))
    shard_paths: list[str] = []
    shard_sizes: list[int] = []
    for shard_index in range(args.shards):
        shard = videos[shard_index :: args.shards]
        shard_path = output / (
            f"{args.prefix}_{shard_index + 1:0{widths}d}.{args.format}"
        )
        serialized = [str(video) for video in shard]
        if args.format == "txt":
            content = "".join(f"{video}\n" for video in serialized)
        else:
            content = json.dumps(
                {
                    "format_version": 1,
                    "shard_index": shard_index + 1,
                    "shard_count": args.shards,
                    "videos": serialized,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n"
        shard_path.write_text(content, encoding="utf-8")
        shard_paths.append(str(shard_path))
        shard_sizes.append(len(shard))
    print(
        json.dumps(
            {
                "input_dir": str(source),
                "videos_found": len(videos),
                "shard_count": args.shards,
                "shard_sizes": shard_sizes,
                "shards": shard_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
