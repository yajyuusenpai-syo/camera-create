"""Launch Pi3X and MoGe-3 in isolated Python interpreters and validate their caches."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
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
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_USER"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        subprocess.run(args, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{model} worker failed with exit code {exc.returncode}") from exc


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
) -> DepthWorkerResult:
    """Execute Pi3X using only its isolated interpreter."""
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
) -> DepthWorkerResult:
    """Execute MoGe-3 using only its isolated interpreter."""
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
