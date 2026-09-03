"""Test that missing VIPE source integration is rejected before model inference."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_create.vipe_runner import preflight_vipe_integration


def test_preflight_rejects_missing_frame_index_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vipe = bin_dir / "vipe"
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    vipe.touch()
    python.touch()
    monkeypatch.setattr("camera_create.vipe_runner.find_vipe", lambda _command: str(vipe))
    monkeypatch.setattr(
        "camera_create.vipe_runner.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="setup_vipe.py"):
        preflight_vipe_integration(str(vipe))
