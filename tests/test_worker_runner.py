"""Unit tests for cross-environment depth-cache validation."""

import os
from pathlib import Path

import numpy as np
import pytest

from camera_create.worker_runner import (
    _require_executable,
    ensure_matching_workers,
    load_worker_cache,
    run_pi3x_worker,
)


def write_cache(path: Path, frames: int = 2, width: int = 8) -> None:
    """Create a small valid worker cache without loading a model."""
    np.savez_compressed(
        path,
        depth=np.ones((frames, 4, width), dtype=np.float32),
        frame_count=frames,
        original_width=16,
        original_height=8,
        inference_width=width,
        inference_height=4,
        fps=24.0,
    )


def test_worker_cache_contract_and_match(tmp_path: Path) -> None:
    pi3_path = tmp_path / "pi3.npz"
    moge_path = tmp_path / "moge.npz"
    write_cache(pi3_path)
    write_cache(moge_path)
    pi3 = load_worker_cache(pi3_path, "Pi3X")
    moge = load_worker_cache(moge_path, "MoGe-3")
    ensure_matching_workers(pi3, moge)
    assert pi3.depth.shape == (2, 4, 8)


def test_worker_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    pi3_path = tmp_path / "pi3.npz"
    moge_path = tmp_path / "moge.npz"
    write_cache(pi3_path)
    write_cache(moge_path, frames=3)
    with pytest.raises(ValueError, match="metadata mismatch"):
        ensure_matching_workers(
            load_worker_cache(pi3_path, "Pi3X"),
            load_worker_cache(moge_path, "MoGe-3"),
        )


def test_pi3x_worker_uses_selected_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "pi3-python"
    executable.touch()
    output = tmp_path / "result.npz"
    invoked: list[str] = []

    def fake_run(args: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        assert env["PYTHONNOUSERSITE"] == "1"
        assert "PYTHONPATH" not in env
        invoked.extend(args)
        write_cache(output)

    monkeypatch.setattr("camera_create.worker_runner.subprocess.run", fake_run)
    result = run_pi3x_worker(
        executable,
        tmp_path / "input.mp4",
        tmp_path / "ckpt",
        output,
        "cuda:0",
        8,
        4,
        448,
        True,
        True,
    )
    assert Path(invoked[0]) == executable.absolute()
    assert Path(invoked[1]).name == "run_pi3x_worker.py"
    assert "--disable-cudnn" in invoked
    assert "--disable-sdp" in invoked
    assert result.frame_count == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv Python uses symlinks")
def test_venv_python_symlink_is_not_resolved(tmp_path: Path) -> None:
    system_python = tmp_path / "usr" / "bin" / "python3.10"
    system_python.parent.mkdir(parents=True)
    system_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)

    selected = _require_executable(venv_python, "Pi3X")

    assert selected == venv_python.absolute()
    assert selected != system_python.absolute()
