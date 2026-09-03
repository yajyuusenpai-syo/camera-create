"""Test deterministic VIPE cache paths and fail-fast offline validation."""

from pathlib import Path

import pytest

from camera_create.vipe_assets import asset_paths, missing_assets, require_assets


def test_asset_paths_match_upstream_torch_hub_layout(tmp_path: Path) -> None:
    paths = asset_paths(tmp_path)
    assert paths["DROID-SLAM"] == tmp_path / "hub/droid_slam/droid.pth"
    assert paths["GeoCalib pinhole"] == tmp_path / "hub/geocalib/pinhole.tar"


def test_require_assets_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DROID-SLAM"):
        require_assets(tmp_path)


def test_nonempty_assets_are_ready(tmp_path: Path) -> None:
    for path in asset_paths(tmp_path).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    assert missing_assets(tmp_path) == {}
    require_assets(tmp_path)
