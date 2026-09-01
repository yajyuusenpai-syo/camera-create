"""VIPE depth backend that reads camera_create's precomputed metric NPZ cache."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import (
    DepthEstimationInput,
    DepthEstimationModel,
    DepthEstimationResult,
    DepthType,
)


class CachedDepthModel(DepthEstimationModel):
    """Look up metric depth using VIPE's raw video frame index."""

    def __init__(self, cache_path: str):
        with np.load(cache_path) as data:
            self._depths = data["depths"].astype(np.float32)

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        index = src.frame_idx
        if index is None or not 0 <= index < len(self._depths):
            raise ValueError(
                f"Invalid cached-depth frame index {index}; cache length={len(self._depths)}"
            )
        if src.rgb is None:
            raise ValueError(
                "CachedDepthModel requires RGB to determine the target resolution"
            )
        target_height, target_width = int(src.rgb.shape[-3]), int(src.rgb.shape[-2])
        depth = torch.from_numpy(self._depths[index])[None, None]
        if tuple(depth.shape[-2:]) != (target_height, target_width):
            depth = F.interpolate(
                depth,
                (target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        return DepthEstimationResult(metric_depth=depth.squeeze(0))
