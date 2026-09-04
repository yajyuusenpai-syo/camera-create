"""Test recursive discovery, static worker assignment, resume, and camera JSON v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camera_create.artifacts import export_camera_json_v2
from camera_create.batch import (
    assign_tasks,
    camera_artifact_dir,
    camera_json_path,
    discover_videos,
    prepare_video,
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
    assert camera_artifact_dir(first).name == "a.MP4.camera"
    assert camera_artifact_dir(second).parent == nested.resolve()


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
    payload = export_camera_json_v2(
        result, output, video.name, 30.0, 24.0, 241, 10.06
    )
    assert payload["format_version"] == 2
    assert payload["is_metric"] is True
    assert payload["max_frames"] == 241
    assert payload["frames"][1]["timestamp_seconds"] == 1 / 24
    assert payload["frames"][1]["c2w"][0][3] == 1.25
    assert payload["frames"][0]["intrinsics"][0][0] == 500
    assert valid_existing_output(video)
    assert valid_existing_output(video, max_frames=241)
    assert not valid_existing_output(video, max_frames=120)
    assert not valid_existing_output(video, target_fps=30.0)
    assert not valid_existing_output(video, max_video_seconds=5.0)


def test_prepare_video_applies_fps_frame_and_duration_limits(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mkv"
    source.touch()
    commands: list[list[str]] = []

    def fake_probe(path: Path) -> tuple[float, int, float]:
        return (30.0, 300, 10.0) if path == source else (24.0, 241, 241 / 24)

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        Path(command[-1]).touch()

    monkeypatch.setattr("camera_create.batch._probe_video", fake_probe)
    monkeypatch.setattr("camera_create.batch.shutil.which", lambda _name: "/bin/ffmpeg")
    monkeypatch.setattr("camera_create.batch.subprocess.run", fake_run)

    processed, source_fps, max_seconds = prepare_video(
        source, tmp_path / "work", 24.0, 241, 10.06, "ffmpeg"
    )

    assert processed.is_file()
    assert source_fps == 30.0
    assert max_seconds == 10.06
    assert commands[0][commands[0].index("-vf") + 1] == "fps=24.0"
    assert commands[0][commands[0].index("-frames:v") + 1] == "241"
    assert commands[0][commands[0].index("-t") + 1] == "10.06"
