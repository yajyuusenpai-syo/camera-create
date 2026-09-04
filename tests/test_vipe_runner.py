"""Test that missing VIPE source integration is rejected before model inference."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_create.vipe_runner import preflight_vipe_integration, run_vipe


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


def test_vipe_process_receives_global_backend_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vipe = bin_dir / "vipe"
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    vipe.touch()
    python.touch()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(args, check, env):
        assert check
        calls.append((args, env))

    monkeypatch.setattr("camera_create.vipe_runner.find_vipe", lambda _command: str(vipe))
    monkeypatch.setattr("camera_create.vipe_runner.subprocess.run", fake_run)
    monkeypatch.setattr("camera_create.vipe_runner.preflight_vipe_assets", lambda *_args: None)

    run_vipe(
        tmp_path / "video.mp4",
        tmp_path / "output",
        tmp_path / "metric.npz",
        tmp_path / "model-cache",
        str(vipe),
        allow_downloads=True,
        disable_cudnn=True,
        disable_sdp=True,
    )

    args, environment = calls[0]
    assert Path(args[0]) == python.resolve()
    assert args[1] == "-c"
    assert args[3] == "infer"
    assert environment["CAMERA_CREATE_DISABLE_CUDNN"] == "1"
    assert environment["CAMERA_CREATE_DISABLE_SDP"] == "1"
