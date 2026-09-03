"""Locate and validate the external DROID-SLAM and GeoCalib assets used by VIPE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VipeAsset:
    """Describe one upstream VIPE asset and its path below TORCH_HOME."""

    name: str
    relative_path: Path
    source_url: str
    minimum_bytes: int


VIPE_ASSETS = (
    VipeAsset(
        name="DROID-SLAM",
        relative_path=Path("hub/droid_slam/droid.pth"),
        source_url=(
            "https://huggingface.co/vslamlab/droidslam/resolve/main/droid.pth"
        ),
        minimum_bytes=15_000_000,
    ),
    VipeAsset(
        name="GeoCalib pinhole",
        relative_path=Path("hub/geocalib/pinhole.tar"),
        source_url=(
            "https://github.com/cvg/GeoCalib/releases/download/v1.0/"
            "geocalib-pinhole.tar"
        ),
        minimum_bytes=1_000_000,
    ),
)


def asset_paths(torch_home: Path) -> dict[str, Path]:
    """Return the exact paths where pinned VIPE v1.2.0 looks for its assets."""
    return {
        asset.name: torch_home.resolve() / asset.relative_path
        for asset in VIPE_ASSETS
    }


def missing_assets(torch_home: Path) -> dict[str, Path]:
    """Return missing or empty VIPE assets under a selected TORCH_HOME."""
    paths = asset_paths(torch_home)
    return {
        asset.name: paths[asset.name]
        for asset in VIPE_ASSETS
        if not paths[asset.name].is_file()
        or paths[asset.name].stat().st_size < asset.minimum_bytes
    }


def require_assets(torch_home: Path) -> None:
    """Fail before inference instead of allowing an unexpected network download."""
    missing = missing_assets(torch_home)
    if not missing:
        return
    details = "\n".join(f"  - {name}: {path}" for name, path in missing.items())
    raise FileNotFoundError(
        "VIPE runtime assets are not available locally:\n"
        f"{details}\n"
        "Run scripts/prepare_vipe_assets.py on a connected machine, or copy "
        "ckpt/vipe/torch from one. Pass --allow-vipe-downloads only when this "
        "machine can reach Google Drive and GitHub."
    )
