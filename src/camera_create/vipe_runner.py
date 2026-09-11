"""Invoke patched NVIDIA VIPE with the fused metric-depth cache."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path

from .vipe_assets import require_assets

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VipeServiceEndpoint:
    """Connection details for one persistent VIPE process."""

    host: str
    port: int
    authkey_hex: str


@dataclass
class VipeServiceProcess:
    """Own one persistent VIPE subprocess and its local endpoint."""

    endpoint: VipeServiceEndpoint
    process: subprocess.Popen

    def stop(self) -> None:
        """Terminate the service without blocking behind an active inference."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

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


def _clean_vipe_environment(
    model_cache: Path, disable_cudnn: bool, disable_sdp: bool
) -> dict[str, str]:
    """Build an isolated environment shared by one-shot and service modes."""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_USER"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HOME"] = str((model_cache / "huggingface").resolve())
    environment["TORCH_HOME"] = str((model_cache / "torch").resolve())
    if disable_cudnn:
        environment["CAMERA_CREATE_DISABLE_CUDNN"] = "1"
    if disable_sdp:
        environment["CAMERA_CREATE_DISABLE_SDP"] = "1"
    return environment


def start_vipe_service(
    command: str,
    model_cache: Path,
    gpu_id: int,
    ready_file: Path,
    *,
    allow_downloads: bool = False,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
    recycle_every: int = 25,
    assets_preflight_done: bool = False,
    timeout_seconds: float = 1800.0,
) -> VipeServiceProcess:
    """Start VIPE once, retain its stateless models, and expose a local endpoint."""
    if not assets_preflight_done:
        preflight_vipe_assets(model_cache, allow_downloads)
    executable = Path(find_vipe(command)).resolve()
    python = executable.parent / ("python.exe" if os.name == "nt" else "python")
    if not python.is_file():
        raise RuntimeError(f"Cannot locate VIPE Python beside {executable}: {python}")
    token = secrets.token_hex(32)
    args = [
        str(python),
        str(PROJECT_ROOT / "scripts" / "run_vipe_worker.py"),
        "--ready-file", str(ready_file),
        "--authkey", token,
        "--recycle-every", str(recycle_every),
    ]
    environment = _clean_vipe_environment(model_cache, disable_cudnn, disable_sdp)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.unlink(missing_ok=True)
    process = subprocess.Popen(args, env=environment)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Persistent VIPE service exited during startup with code {process.returncode}"
            )
        if ready_file.is_file():
            try:
                state = json.loads(ready_file.read_text(encoding="utf-8"))
                endpoint = VipeServiceEndpoint(
                    str(state["host"]), int(state["port"]), token
                )
                return VipeServiceProcess(endpoint, process)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(0.2)
    process.terminate()
    raise TimeoutError("Timed out loading persistent VIPE service")


def _request_vipe_service(
    endpoint: VipeServiceEndpoint, video: Path, output_dir: Path, cache_path: Path
) -> None:
    """Run one video synchronously through an already initialized VIPE service."""
    connection = Client(
        (endpoint.host, endpoint.port),
        family="AF_INET",
        authkey=bytes.fromhex(endpoint.authkey_hex),
    )
    try:
        connection.send(
            {
                "command": "infer",
                "input": str(video),
                "output": str(output_dir),
                "cache_path": str(cache_path),
            }
        )
        response = connection.recv()
    finally:
        connection.close()
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error", response) if isinstance(response, dict) else response
        raise RuntimeError(f"Persistent VIPE worker failed: {detail}")


def run_vipe(
    video: Path,
    output_dir: Path,
    cache_path: Path,
    model_cache: Path,
    command: str = "vipe",
    allow_downloads: bool = False,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
    service: VipeServiceEndpoint | None = None,
    assets_preflight_done: bool = False,
) -> None:
    """Run VIPE cached-depth BA, inheriting metric scale from the depth cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if service is not None:
        _request_vipe_service(service, video, output_dir, cache_path)
        return
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
    if not assets_preflight_done:
        preflight_vipe_assets(model_cache, allow_downloads)
    process_env = _clean_vipe_environment(model_cache, disable_cudnn, disable_sdp)
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
