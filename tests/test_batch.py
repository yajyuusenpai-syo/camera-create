"""Test recursive discovery, static worker assignment, resume, and camera JSON v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camera_create.artifacts import export_camera_json_v2
from camera_create.batch import (
    assign_tasks,
    camera_json_path,
    discover_videos,
    valid_existing_output,
)


def test_recursive_discovery_and_static_assignment(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.MP4"
    second = nested / "b.mkv"
    ignored = nested / "notes.txt"
    for path in (first, second, ignored):
        path.touch()
    videos = discover_videos(tmp_path, (".mp4", ".mkv"))
    assert videos == [first.resolve(), second.resolve()]
    assert assign_tasks(videos, 2) == [[first.resolve()], [second.resolve()]]
    assert camera_json_path(first).name == "cam_a.MP4.json"


def test_export_and_resume_metric_json_v2(tmp_path: Path) -> None:
    result = tmp_path / "result"
    result.mkdir()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = 1.25
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 500
    intrinsics[:, 1, 1] = 490
    np.save(result / "poses_c2w_metric.npy", poses)
    np.save(result / "intrinsics_K.npy", intrinsics)
    (result / "camera_report.json").write_text(
        json.dumps({"valid": True}), encoding="utf-8"
    )
    video = tmp_path / "clip.mp4"
    video.touch()
    output = camera_json_path(video)
    payload = export_camera_json_v2(result, output, video.name, 30.0, 24.0, 10.06)
    assert payload["format_version"] == 2
    assert payload["is_metric"] is True
    assert payload["frames"][1]["timestamp_seconds"] == 1 / 24
    assert payload["frames"][1]["c2w"][0][3] == 1.25
    assert payload["frames"][0]["intrinsics"][0][0] == 500
    assert valid_existing_output(video)
    assert not valid_existing_output(video, target_fps=30.0)
    assert not valid_existing_output(video, max_video_seconds=5.0)
