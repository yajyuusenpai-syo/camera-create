"""Invoke patched NVIDIA VIPE with the fused metric-depth cache."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .vipe_assets import require_assets


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


@contextmanager
def _model_cache_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set model-cache locations only when the caller has not configured them."""
    inserted: list[str] = []
    for name, value in values.items():
        if name not in os.environ:
            os.environ[name] = value
            inserted.append(name)
    try:
        yield
    finally:
        for name in inserted:
            os.environ.pop(name, None)


def find_vipe(command: str = "vipe") -> str:
    """Resolve VIPE and return a helpful installation error when absent."""
    resolved = shutil.which(command)
    if resolved is None:
        raise RuntimeError(
            "VIPE executable not found. Run camera_create/scripts/setup_vipe.py first."
        )
    return resolved


def vipe_torch_home(model_cache: Path) -> Path:
    """Resolve the Torch Hub cache exactly as the isolated VIPE process will."""
    return Path(
        os.environ.get("TORCH_HOME", str((model_cache / "torch").resolve()))
    ).resolve()


def preflight_vipe_assets(model_cache: Path, allow_downloads: bool = False) -> None:
    """Check runtime weights before expensive Pi3X and MoGe-3 inference starts."""
    if not allow_downloads:
        require_assets(vipe_torch_home(model_cache))


def run_vipe(
    video: Path,
    output_dir: Path,
    cache_path: Path,
    model_cache: Path,
    command: str = "vipe",
    allow_downloads: bool = False,
) -> None:
    """Run VIPE cached-depth BA, inheriting metric scale from the depth cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = find_vipe(command)
    args = [
        executable,
        "infer",
        str(video),
        "--output",
        str(output_dir),
        "--pipeline",
        "vipe_cached_depth",
    ]
    model_cache.mkdir(parents=True, exist_ok=True)
    cache_env = {
        "HF_HOME": str((model_cache / "huggingface").resolve()),
        "TORCH_HOME": str((model_cache / "torch").resolve()),
    }
    preflight_vipe_assets(model_cache, allow_downloads)
    process_env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_USER"):
        process_env.pop(name, None)
    process_env["PYTHONNOUSERSITE"] = "1"
    with _model_cache_environment(cache_env), _temporary_environment(
        "SANA_WM_CACHED_DEPTH_PATH", str(cache_path.resolve())
    ):
        process_env.update(
            {
                "HF_HOME": os.environ["HF_HOME"],
                "TORCH_HOME": os.environ["TORCH_HOME"],
                "SANA_WM_CACHED_DEPTH_PATH": os.environ[
                    "SANA_WM_CACHED_DEPTH_PATH"
                ],
            }
        )
        subprocess.run(args, check=True, env=process_env)
