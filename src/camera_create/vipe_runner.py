"""Invoke patched NVIDIA VIPE with the fused metric-depth cache."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .vipe_assets import require_assets

VIPE_BACKEND_BOOTSTRAP = """
import os
import torch
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
from vipe.cli.main import main
main()
"""


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


def preflight_vipe_integration(command: str = "vipe") -> None:
    """Verify v1.2 has the cached-depth frame-index backport before GPU work."""
    executable = Path(find_vipe(command)).resolve()
    python = executable.parent / ("python.exe" if os.name == "nt" else "python")
    if not python.is_file():
        raise RuntimeError(
            f"Cannot locate the VIPE environment Python beside {executable}: {python}"
        )
    source = (
        "from pathlib import Path; import vipe; "
        "from vipe.priors.depth.base import DepthEstimationInput; "
        "root=Path(vipe.__file__).resolve().parent; "
        "buffer=(root/'slam/components/buffer.py').read_text(); "
        "assert 'frame_idx' in DepthEstimationInput.__dataclass_fields__; "
        "assert 'frame_idx=int(self.tstamp[frame_idx].item())' in buffer"
    )
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_USER"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(python), "-c", source],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "VIPE cached-depth frame-index patch is missing. Before inference run:\n"
            "  .envs/vipe/bin/python scripts/setup_vipe.py "
            "--vipe-source third_party/vipe --skip-install"
        )


def run_vipe(
    video: Path,
    output_dir: Path,
    cache_path: Path,
    model_cache: Path,
    command: str = "vipe",
    allow_downloads: bool = False,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
) -> None:
    """Run VIPE cached-depth BA, inheriting metric scale from the depth cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = Path(find_vipe(command)).resolve()
    python = executable.parent / ("python.exe" if os.name == "nt" else "python")
    if not python.is_file():
        raise RuntimeError(
            f"Cannot locate the VIPE environment Python beside {executable}: {python}"
        )
    args = [
        str(python),
        "-c",
        VIPE_BACKEND_BOOTSTRAP,
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
    if disable_cudnn:
        process_env["CAMERA_CREATE_DISABLE_CUDNN"] = "1"
    if disable_sdp:
        process_env["CAMERA_CREATE_DISABLE_SDP"] = "1"
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
