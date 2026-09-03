"""Convert sparse VIPE artifacts into dense, validated metric camera arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def interpolate_poses(
    poses: np.ndarray, indices: np.ndarray, frame_count: int
) -> np.ndarray:
    """Densify c2w poses using linear translation and SO(3) quaternion SLERP."""
    order = np.argsort(indices)
    indices = np.asarray(indices[order], dtype=np.int64)
    poses = np.asarray(poses[order], dtype=np.float64)
    if len(indices) == 0 or indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError("VIPE pose indices are empty or outside the video")
    unique, positions = np.unique(indices, return_index=True)
    indices, poses = unique, poses[positions]
    rotations = np.stack([_project_rotation(pose[:3, :3]) for pose in poses])
    query = np.arange(frame_count, dtype=np.float64)
    clipped = np.clip(query, indices[0], indices[-1])
    if len(indices) == 1:
        dense_rotations = np.repeat(rotations, frame_count, axis=0)
    else:
        dense_rotations = Slerp(indices.astype(float), Rotation.from_matrix(rotations))(
            clipped
        ).as_matrix()
    translations = np.stack(
        [np.interp(query, indices, poses[:, axis, 3]) for axis in range(3)], axis=1
    )
    dense = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    dense[:, :3, :3] = dense_rotations
    dense[:, :3, 3] = translations
    # Express all cameras in the first-camera world frame.
    dense = np.linalg.inv(dense[0])[None] @ dense
    return dense.astype(np.float32)


def interpolate_intrinsics(
    values: np.ndarray, indices: np.ndarray, frame_count: int
) -> np.ndarray:
    """Linearly densify pixel intrinsics [fx, fy, cx, cy]."""
    if len(indices) == 0:
        raise ValueError("VIPE intrinsics artifact is empty")
    order = np.argsort(indices)
    indices = np.asarray(indices[order], dtype=np.int64)
    values = np.asarray(values[order], dtype=np.float64)
    unique, positions = np.unique(indices, return_index=True)
    indices, values = unique, values[positions]
    query = np.arange(frame_count)
    dense = np.stack(
        [np.interp(query, indices, values[:, axis]) for axis in range(4)], axis=1
    )
    return dense.astype(np.float32)


def intrinsics_to_k(intrinsics: np.ndarray) -> np.ndarray:
    """Convert [fx, fy, cx, cy] arrays to 3x3 calibration matrices."""
    matrices = np.zeros((len(intrinsics), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = intrinsics[:, 0]
    matrices[:, 1, 1] = intrinsics[:, 1]
    matrices[:, 0, 2] = intrinsics[:, 2]
    matrices[:, 1, 2] = intrinsics[:, 3]
    matrices[:, 2, 2] = 1.0
    return matrices


def _load_vipe_npz(
    video: Path, vipe_dir: Path, category: str
) -> tuple[np.ndarray, np.ndarray]:
    path = vipe_dir / category / f"{video.stem}.npz"
    if not path.exists():
        raise FileNotFoundError(f"VIPE did not produce {category} artifact: {path}")
    with np.load(path) as data:
        return data["data"].copy(), data["inds"].copy()


def validate_camera(
    poses_c2w: np.ndarray, intrinsics: np.ndarray, scale: np.ndarray
) -> dict:
    """Return numerical checks that establish the output as usable camera data."""
    rotations = poses_c2w[:, :3, :3].astype(np.float64)
    identities = np.eye(3)[None]
    orthogonality = np.linalg.norm(
        np.swapaxes(rotations, 1, 2) @ rotations - identities, axis=(1, 2)
    )
    determinants = np.linalg.det(rotations)
    translation_steps = np.linalg.norm(np.diff(poses_c2w[:, :3, 3], axis=0), axis=1)
    checks = {
        "all_finite": bool(
            np.isfinite(poses_c2w).all()
            and np.isfinite(intrinsics).all()
            and np.isfinite(scale).all()
        ),
        "positive_focal_lengths": bool((intrinsics[:, :2] > 0).all()),
        "first_pose_identity": bool(np.allclose(poses_c2w[0], np.eye(4), atol=1e-4)),
        "max_rotation_orthogonality_error": float(orthogonality.max()),
        "max_rotation_determinant_error": float(np.max(np.abs(determinants - 1.0))),
        "scale_min": float(np.nanmin(scale)),
        "scale_max": float(np.nanmax(scale)),
        "scale_cv": float(np.nanstd(scale) / max(abs(np.nanmean(scale)), 1e-8)),
        "trajectory_length_m": float(translation_steps.sum()),
    }
    checks["valid"] = bool(
        checks["all_finite"]
        and checks["positive_focal_lengths"]
        and checks["first_pose_identity"]
        and checks["max_rotation_orthogonality_error"] < 1e-3
        and checks["max_rotation_determinant_error"] < 1e-3
    )
    return checks


def export_camera_artifacts(
    video: Path,
    vipe_dir: Path,
    output_dir: Path,
    frame_count: int,
    scale_history: np.ndarray,
    metadata: dict,
) -> dict:
    """Load VIPE output and write the public camera_create output contract."""
    sparse_poses, pose_indices = _load_vipe_npz(video, vipe_dir, "pose")
    sparse_intrinsics, intr_indices = _load_vipe_npz(video, vipe_dir, "intrinsics")
    poses_c2w = interpolate_poses(sparse_poses, pose_indices, frame_count)
    intrinsics = interpolate_intrinsics(
        sparse_intrinsics[:, :4], intr_indices, frame_count
    )
    poses_w2c = np.linalg.inv(poses_c2w).astype(np.float32)
    k_matrices = intrinsics_to_k(intrinsics)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "poses_c2w_metric.npy", poses_c2w)
    np.save(output_dir / "extrinsics_w2c_metric.npy", poses_w2c)
    np.save(output_dir / "intrinsics.npy", intrinsics[:, None, :])
    np.save(output_dir / "intrinsics_K.npy", k_matrices)
    np.save(output_dir / "scale_per_frame.npy", scale_history.astype(np.float32))
    report = validate_camera(poses_c2w, intrinsics, scale_history)
    report.update(metadata)
    (output_dir / "camera_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not report["valid"]:
        raise RuntimeError(
            f"Camera output failed validation; inspect {output_dir / 'camera_report.json'}"
        )
    return report


def export_camera_json_v2(
    result_dir: Path,
    output_path: Path,
    video_name: str,
    source_fps: float,
    target_fps: float,
    max_video_seconds: float,
) -> dict:
    """Convert validated metric NPY artifacts into the per-frame JSON v2 contract."""
    poses_path = result_dir / "poses_c2w_metric.npy"
    intrinsics_path = result_dir / "intrinsics_K.npy"
    report_path = result_dir / "camera_report.json"
    if not poses_path.is_file() or not intrinsics_path.is_file():
        raise FileNotFoundError(f"Metric camera artifacts are incomplete: {result_dir}")
    poses = np.load(poses_path, allow_pickle=False)
    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    if poses.shape != (len(poses), 4, 4):
        raise ValueError(f"Unexpected c2w shape: {poses.shape}")
    if intrinsics.shape != (len(poses), 3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if report_path.is_file():
        validation = json.loads(report_path.read_text(encoding="utf-8"))
        if not validation.get("valid", False):
            raise RuntimeError(
                f"Refusing to export invalid camera result: {report_path}"
            )
    payload = {
        "format_version": 2,
        "video_name": video_name,
        "fps": float(target_fps),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "frame_count": len(poses),
        "is_metric": True,
        "max_video_seconds": float(max_video_seconds),
        "camera_convention": "OpenCV: +X right, +Y down, +Z forward",
        "frames": [
            {
                "frame_index": index,
                "timestamp_seconds": index / float(target_fps),
                "c2w": poses[index].astype(float).tolist(),
                "intrinsics": intrinsics[index].astype(float).tolist(),
            }
            for index in range(len(poses))
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload
