"""Run Pi3X inference helpers and fuse Pi3X/MoGe-3 depths for VIPE."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)
MIN_DEPTH = 1e-3


def _window_starts(frame_count: int, chunk: int, stride: int) -> list[int]:
    if chunk < 1 or stride < 1:
        raise ValueError("chunk and stride must be positive")
    starts = list(range(0, max(frame_count - chunk + 1, 1), stride))
    tail = max(0, frame_count - chunk)
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return sorted(set(starts))


def load_pi3x(checkpoint: Path, device: str):
    """Load Pi3X using the same API as the SANA-WM Stage 2 implementation."""
    try:
        from pi3 import Pi3X  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pi3X package is not installed; see docs/DEPLOYMENT.md"
        ) from exc
    return Pi3X.from_pretrained(str(checkpoint)).to(device).eval()


def infer_pi3x(
    model, frames_rgb: np.ndarray, device: str, chunk: int, stride: int
) -> np.ndarray:
    """Infer overlapping temporal Pi3X windows without placing the full video on GPU."""
    import torch

    frame_count, height, width, _ = frames_rgb.shape
    accum = np.zeros((frame_count, height, width), dtype=np.float32)
    counts = np.zeros(frame_count, dtype=np.float32)
    starts = _window_starts(frame_count, min(chunk, frame_count), stride)
    with torch.inference_mode():
        for number, start in enumerate(starts, 1):
            end = min(start + chunk, frame_count)
            LOG.info("Pi3X window %d/%d: [%d, %d)", number, len(starts), start, end)
            batch = torch.from_numpy(frames_rgb[start:end]).permute(0, 3, 1, 2)
            batch = (
                batch.to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)
            )
            output = model(batch)
            depth = (
                output["local_points"][0, : end - start, ..., 2].float().cpu().numpy()
            )
            accum[start:end] += depth
            counts[start:end] += 1.0
            del batch, output
    return accum / np.maximum(counts[:, None, None], 1.0)


def fuse_metric_depths(
    pi3x_depth: np.ndarray,
    moge3_depth: np.ndarray,
    momentum: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply inverse-depth weighted scale recovery followed by temporal EMA."""
    if pi3x_depth.shape != moge3_depth.shape:
        raise ValueError(
            f"Depth shape mismatch: {pi3x_depth.shape} vs {moge3_depth.shape}"
        )
    if not 0.0 <= momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")
    raw = np.full(len(pi3x_depth), np.nan, dtype=np.float32)
    smooth = np.full(len(pi3x_depth), np.nan, dtype=np.float32)
    previous: float | None = None
    for index, (consistent, metric) in enumerate(zip(pi3x_depth, moge3_depth)):
        valid = (
            np.isfinite(consistent)
            & np.isfinite(metric)
            & (consistent > MIN_DEPTH)
            & (metric > MIN_DEPTH)
        )
        if np.count_nonzero(valid) >= 32:
            # w=1/d_pi3x in weighted LS reduces exactly to sum(metric)/sum(pi3x).
            estimate = float(
                np.sum(metric[valid], dtype=np.float64)
                / np.sum(consistent[valid], dtype=np.float64)
            )
            raw[index] = estimate
            previous = (
                estimate
                if previous is None
                else momentum * previous + (1.0 - momentum) * estimate
            )
        elif previous is None:
            raise RuntimeError(
                f"No valid Pi3X/MoGe-3 overlap in first usable frame (frame {index})"
            )
        smooth[index] = previous
    fused = pi3x_depth * smooth[:, None, None]
    # VIPE expects a positive metric-depth map. Invalid Pi3X pixels are given a
    # conservative floor instead of leaking negative/NaN values into BA.
    fused = np.where(np.isfinite(fused) & (fused > MIN_DEPTH), fused, MIN_DEPTH)
    return fused.astype(np.float32), smooth, raw


def default_fov_x(width: int, focal_guess_px: float | None = None) -> float:
    """Convert a focal guess to horizontal FOV; default to a neutral 60 degrees."""
    if focal_guess_px is None:
        return 60.0
    return math.degrees(2.0 * math.atan(width / (2.0 * focal_guess_px)))


def save_depth_cache(
    path: Path, depths: np.ndarray, scale: np.ndarray, raw_scale: np.ndarray
) -> None:
    """Write the cache contract consumed by the patched VIPE CachedDepthModel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depths=depths, scale_history=scale, raw_scale=raw_scale)
