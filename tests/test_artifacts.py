"""Unit tests for rigid SE(3) interpolation and camera validation."""

import numpy as np
from scipy.spatial.transform import Rotation

from camera_create.artifacts import interpolate_poses, validate_camera


def test_pose_interpolation_stays_on_so3() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    poses[1, 0, 3] = 2.0
    dense = interpolate_poses(poses, np.array([0, 2]), 3)
    assert np.allclose(dense[1, :3, 3], [1, 0, 0])
    assert np.allclose(dense[1, :3, :3].T @ dense[1, :3, :3], np.eye(3), atol=1e-6)
    report = validate_camera(dense, np.tile([500, 500, 320, 240], (3, 1)), np.ones(3))
    assert report["valid"]
