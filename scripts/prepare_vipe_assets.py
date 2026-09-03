#!/usr/bin/env python3
"""Download or import VIPE's DROID-SLAM and GeoCalib weights into its offline cache."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_create.vipe_assets import (
    VIPE_ASSETS,
    asset_paths,
    missing_assets,
)


def _copy_atomically(source: Path, destination: Path) -> None:
    """Copy a local asset without leaving a partial destination after interruption."""
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Asset is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download(url: str, destination: Path) -> None:
    """Download one asset to a temporary file and publish it atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n        -> {destination}", flush=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Populate exact TORCH_HOME paths from local files or known download URLs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "ckpt/vipe")
    parser.add_argument("--droid-source", type=Path)
    parser.add_argument("--geocalib-source", type=Path)
    parser.add_argument(
        "--droid-url",
        default=VIPE_ASSETS[0].source_url,
        help="Override the DROID mirror URL, for example an internal/HF endpoint",
    )
    parser.add_argument(
        "--geocalib-url",
        default=VIPE_ASSETS[1].source_url,
        help="Override the official GeoCalib URL with an internal mirror",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Fetch missing files; the DROID URL is a community Hugging Face mirror",
    )
    args = parser.parse_args()
    torch_home = args.cache_root.resolve() / "torch"
    sources = {
        "DROID-SLAM": args.droid_source,
        "GeoCalib pinhole": args.geocalib_source,
    }
    urls = {
        "DROID-SLAM": args.droid_url,
        "GeoCalib pinhole": args.geocalib_url,
    }
    destinations = asset_paths(torch_home)
    for asset in VIPE_ASSETS:
        destination = destinations[asset.name]
        source = sources[asset.name]
        if source is not None:
            _copy_atomically(source.resolve(), destination)
        elif args.download_missing and (
            not destination.is_file() or destination.stat().st_size == 0
        ):
            _download(urls[asset.name], destination)

    missing = missing_assets(torch_home)
    for name, destination in destinations.items():
        state = "MISSING" if name in missing else "OK"
        size = destination.stat().st_size if destination.is_file() else 0
        print(f"[{state}] {name}: {destination} ({size} bytes)")
    if missing:
        print("Provide the missing local --*-source file or use --download-missing.")
        return 1
    print(f"VIPE offline cache is ready: {torch_home}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
