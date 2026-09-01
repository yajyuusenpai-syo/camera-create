"""Unit tests for metric scale fusion without loading any neural network."""

import numpy as np

from camera_create.depth import fuse_metric_depths


def test_metric_scale_recovery() -> None:
    consistent = np.ones((3, 8, 8), dtype=np.float32) * 2.0
    metric = consistent * 3.0
    fused, smooth, raw = fuse_metric_depths(consistent, metric)
    assert np.allclose(raw, 3.0)
    assert np.allclose(smooth, 3.0)
    assert np.allclose(fused, metric)
