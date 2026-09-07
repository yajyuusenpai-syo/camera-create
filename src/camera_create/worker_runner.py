"""Launch Pi3X and MoGe-3 in isolated Python interpreters and validate their caches."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DepthWorkerResult:
    """Validated depth output and video metadata produced by one model worker."""

    depth: np.ndarray
    frame_count: int
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    fps: float


@dataclass(frozen=True)
class DepthServiceEndpoint:
    """Connection details for one persistent model process in an isolated env."""

    host: str
    port: int
    authkey_hex: str


@dataclass
class DepthServiceProcess:
    """Own one persistent isolated model subprocess and its local endpoint."""

    model: str
    endpoint: DepthServiceEndpoint
    process: subprocess.Popen

    def stop(self) -> None:
        """Terminate the isolated server without waiting behind a stuck request."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def default_environment_executable(environment: str, executable: str = "python") -> Path:
    """Return the project-local venv executable path for Linux or Windows."""
    env = PROJECT_ROOT / ".envs" / environment
    if sys.platform == "win32":
        suffix = ".exe" if executable in {"python", "vipe"} else ""
        return env / "Scripts" / f"{executable}{suffix}"
    return env / "bin" / executable


def _require_executable(path: Path, model: str) -> Path:
    # Do not use Path.resolve() here. A POSIX venv's bin/python is commonly a
    # symlink to the system interpreter; resolving it discards the venv path,
    # so Python no longer discovers the adjacent pyvenv.cfg.
    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_file():
        raise FileNotFoundError(
            f"{model} environment executable not found: {absolute}. "
            "Run scripts/setup_three_envs.sh or pass the matching CLI option."
        )
    return absolute


def _run(args: list[str], model: str) -> None:
    environment = _clean_environment()

    try:
        subprocess.run(args, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{model} worker failed with exit code {exc.returncode}") from exc


def _clean_environment() -> dict[str, str]:
    """Return an isolated child environment that cannot see user site packages."""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_USER"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _request_service(
    endpoint: DepthServiceEndpoint, request: dict, model: str
) -> dict:
    """Send one synchronous inference request to a persistent depth service."""
    connection = Client(
        (endpoint.host, endpoint.port),
        family="AF_INET",
        authkey=bytes.fromhex(endpoint.authkey_hex),
    )
    try:
        connection.send(request)
        response = connection.recv()
    finally:
        connection.close()
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error", response) if isinstance(response, dict) else response
        raise RuntimeError(f"{model} persistent worker failed: {detail}")
    return response


def start_depth_service(
    model: str,
    python: Path,
    checkpoint: Path,
    gpu_id: int,
    ready_file: Path,
    *,
    chunk: int = 16,
    stride: int = 8,
    max_side: int = 560,
    refine_steps: int = 3,
    use_fp16: bool = True,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
    timeout_seconds: float = 1800.0,
) -> DepthServiceProcess:
    """Start one model server, wait until its weights are loaded, and return it."""
    if model not in {"Pi3X", "MoGe-3"}:
        raise ValueError(f"Unsupported persistent depth model: {model}")
    executable = _require_executable(python, model)
    script_name = "run_pi3x_worker.py" if model == "Pi3X" else "run_moge3_worker.py"
    token = secrets.token_hex(32)
    args = [
        str(executable),
        str(PROJECT_ROOT / "scripts" / script_name),
        "--serve",
        "--ready-file",
        str(ready_file),
        "--authkey",
        token,
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cuda:0",
        "--max-inference-side",
        str(max_side),
    ]
    if model == "Pi3X":
        args.extend(["--chunk", str(chunk), "--stride", str(stride)])
    else:
        args.extend(["--refine-steps", str(refine_steps)])
        if use_fp16:
            args.append("--fp16")
    if disable_cudnn:
        args.append("--disable-cudnn")
    if disable_sdp:
        args.append("--disable-sdp")
    environment = _clean_environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.unlink(missing_ok=True)
    process = subprocess.Popen(args, env=environment)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{model} persistent worker exited during startup with code "
                f"{process.returncode}"
            )
        if ready_file.is_file():
            try:
                state = json.loads(ready_file.read_text(encoding="utf-8"))
                endpoint = DepthServiceEndpoint(
                    host=str(state["host"]),
                    port=int(state["port"]),
                    authkey_hex=token,
                )
                return DepthServiceProcess(model, endpoint, process)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(0.2)
    process.terminate()
    raise TimeoutError(f"Timed out loading {model} persistent worker")


def load_worker_cache(path: Path, model: str) -> DepthWorkerResult:
    """Load one worker NPZ and reject incomplete or inconsistent output."""
    if not path.is_file():
        raise FileNotFoundError(f"{model} worker did not create cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "depth",
            "frame_count",
            "original_width",
            "original_height",
            "inference_width",
            "inference_height",
            "fps",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{model} cache missing fields: {sorted(missing)}")
        depth = np.asarray(data["depth"], dtype=np.float32)
        result = DepthWorkerResult(
            depth=depth,
            frame_count=int(data["frame_count"]),
            original_width=int(data["original_width"]),
            original_height=int(data["original_height"]),
            inference_width=int(data["inference_width"]),
            inference_height=int(data["inference_height"]),
            fps=float(data["fps"]),
        )
    expected = (
        result.frame_count,
        result.inference_height,
        result.inference_width,
    )
    if result.depth.shape != expected:
        raise ValueError(f"{model} depth shape {result.depth.shape} != metadata {expected}")
    if not np.any(np.isfinite(result.depth) & (result.depth > 0)):
        raise ValueError(f"{model} cache contains no positive finite depth")
    return result


def ensure_matching_workers(pi3x: DepthWorkerResult, moge3: DepthWorkerResult) -> None:
    """Require both workers to have decoded the same frames at the same size."""
    pi3_meta = (
        pi3x.frame_count,
        pi3x.original_width,
        pi3x.original_height,
        pi3x.inference_width,
        pi3x.inference_height,
    )
    moge_meta = (
        moge3.frame_count,
        moge3.original_width,
        moge3.original_height,
        moge3.inference_width,
        moge3.inference_height,
    )
    if pi3_meta != moge_meta:
        raise ValueError(f"Pi3X/MoGe-3 worker metadata mismatch: {pi3_meta} vs {moge_meta}")


def run_pi3x_worker(
    python: Path,
    video: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    chunk: int,
    stride: int,
    max_side: int,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
    service: DepthServiceEndpoint | None = None,
) -> DepthWorkerResult:
    """Execute Pi3X using only its isolated interpreter."""
    if service is not None:
        _request_service(
            service,
            {"command": "infer", "input": str(video), "output": str(output)},
            "Pi3X",
        )
        return load_worker_cache(output, "Pi3X")
    executable = _require_executable(python, "Pi3X")
    script = PROJECT_ROOT / "scripts" / "run_pi3x_worker.py"
    args = [
        str(executable), str(script), "--input", str(video), "--output", str(output),
        "--checkpoint", str(checkpoint), "--device", device, "--chunk", str(chunk),
        "--stride", str(stride), "--max-inference-side", str(max_side),
    ]
    if disable_cudnn:
        args.append("--disable-cudnn")
    if disable_sdp:
        args.append("--disable-sdp")
    _run(args, "Pi3X")
    return load_worker_cache(output, "Pi3X")


def run_moge3_worker(
    python: Path,
    video: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    max_side: int,
    fov_x_deg: float,
    refine_steps: int,
    use_fp16: bool,
    disable_cudnn: bool = False,
    disable_sdp: bool = False,
    service: DepthServiceEndpoint | None = None,
) -> DepthWorkerResult:
    """Execute MoGe-3 using only its isolated interpreter."""
    if service is not None:
        _request_service(
            service,
            {
                "command": "infer",
                "input": str(video),
                "output": str(output),
                "fov_x_deg": fov_x_deg,
            },
            "MoGe-3",
        )
        return load_worker_cache(output, "MoGe-3")
    executable = _require_executable(python, "MoGe-3")
    script = PROJECT_ROOT / "scripts" / "run_moge3_worker.py"
    args = [
        str(executable), str(script), "--input", str(video), "--output", str(output),
        "--checkpoint", str(checkpoint), "--device", device,
        "--max-inference-side", str(max_side), "--fov-x-deg", str(fov_x_deg),
        "--refine-steps", str(refine_steps),
    ]
    if use_fp16:
        args.append("--fp16")
    if disable_cudnn:
        args.append("--disable-cudnn")
    if disable_sdp:
        args.append("--disable-sdp")
    _run(args, "MoGe-3")
    return load_worker_cache(output, "MoGe-3")
