"""Unit tests for rigid SE(3) interpolation and camera validation."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camera_create.artifacts import (
    CameraValidationError,
    export_camera_artifacts,
    interpolate_poses,
    validate_camera,
)


def test_pose_interpolation_stays_on_so3() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    poses[1, 0, 3] = 2.0
    dense = interpolate_poses(poses, np.array([0, 2]), 3)
    assert np.allclose(dense[1, :3, 3], [1, 0, 0])
    assert np.allclose(dense[1, :3, :3].T @ dense[1, :3, :3], np.eye(3), atol=1e-6)
    report = validate_camera(dense, np.tile([500, 500, 320, 240], (3, 1)), np.ones(3))
    assert report["valid"]


def test_malformed_vipe_pose_is_a_validation_rejection(tmp_path: Path) -> None:
    video = tmp_path / "bad.mp4"
    vipe_dir = tmp_path / "vipe"
    (vipe_dir / "pose").mkdir(parents=True)
    (vipe_dir / "intrinsics").mkdir(parents=True)
    poses = np.eye(4, dtype=np.float32)[None]
    poses[0, 0, 0] = np.nan
    np.savez(vipe_dir / "pose" / "bad.npz", data=poses, inds=np.array([0]))
    np.savez(
        vipe_dir / "intrinsics" / "bad.npz",
        data=np.array([[500, 500, 320, 240]], dtype=np.float32),
        inds=np.array([0]),
    )
    output = tmp_path / "camera"

    with pytest.raises(CameraValidationError, match="malformed_vipe_camera_output"):
        export_camera_artifacts(
            video,
            vipe_dir,
            output,
            1,
            np.ones(1, dtype=np.float32),
            {"original_width": 640, "original_height": 480},
        )

    report = json.loads((output / "camera_report.json").read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["validation_failure"] == "malformed_vipe_camera_output"
    assert report["validation_error_type"] == "LinAlgError"
