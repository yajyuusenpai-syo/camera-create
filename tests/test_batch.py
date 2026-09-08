"""Test recursive discovery, static worker assignment, resume, and camera JSON v2."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_create.artifacts import export_camera_json_v2, normalize_intrinsics_k
from camera_create.batch import (
    BatchOptions,
    _start_depth_service_group,
    _write_failure_report,
    assign_tasks,
    camera_artifact_dir,
    camera_json_path,
    depth_service_replica_id,
    discover_videos,
    load_video_manifest,
    prepare_video,
    valid_existing_output,
    validate_existing_output_owners,
    validate_unique_camera_outputs,
)
from camera_create.config import ModelPaths
from camera_create.pipeline import PipelineOptions
from camera_create.worker_runner import DepthServiceEndpoint


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
    assert camera_json_path(first).name == "cam_a.json"
    assert camera_artifact_dir(first).name == "a.MP4.camera"
    assert camera_artifact_dir(second).parent == nested.resolve()


def test_failure_report_is_written_beside_shard(tmp_path: Path) -> None:
    manifest = tmp_path / "clip_1.txt"
    manifest.touch()
    run_root = tmp_path / "state" / "run_abc"
    run_root.mkdir(parents=True)
    failed_video = tmp_path / "bad.mp4"
    (run_root / "worker_003.json").write_text(
        json.dumps(
            {
                "global_worker_id": 3,
                "gpu_id": 1,
                "tasks": {
                    str(failed_video): {
                        "status": "failed",
                        "completed_stages": ["pi3x", "moge3", "metric_depth"],
                        "stage_cache": "/cache/bad",
                        "error": "Traceback\nRuntimeError: VIPE failed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report_path, failed_manifest = _write_failure_report(
        manifest, run_root, "abc", 0, 1, 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path == tmp_path / "clip_1.camera_create_failures.json"
    assert failed_manifest == tmp_path / "clip_1.camera_create_failed.txt"
    assert failed_manifest.read_text(encoding="utf-8") == f"{failed_video}\n"
    assert report["failed_count"] == 1
    assert report["failures"][0]["video_path"] == str(failed_video)
    assert report["failures"][0]["likely_failed_stage"] == "vipe"
    assert "RuntimeError: VIPE failed" in report["failures"][0]["error"]


def test_empty_failed_manifest_is_still_created(tmp_path: Path) -> None:
    manifest = tmp_path / "clip_2.txt"
    manifest.touch()
    run_root = tmp_path / "run_empty"
    run_root.mkdir()

    _, failed_manifest = _write_failure_report(manifest, run_root, "empty", 0, 1, 0)

    assert failed_manifest.is_file()
    assert failed_manifest.read_text(encoding="utf-8") == ""


def test_load_txt_and_json_video_manifests(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    first = videos / "a.mp4"
    second = videos / "b.mkv"
    first.touch()
    second.touch()
    text_manifest = tmp_path / "clip_1.txt"
    text_manifest.write_text(
        f"# shard one\nvideos/{first.name}\n{second.resolve()}\n", encoding="utf-8"
    )
    json_manifest = tmp_path / "clip_2.json"
    json_manifest.write_text(
        json.dumps({"videos": [{"path": str(first)}, str(second)]}),
        encoding="utf-8",
    )

    expected = [first.resolve(), second.resolve()]
    assert load_video_manifest(text_manifest, (".mp4", ".mkv")) == expected
    assert load_video_manifest(json_manifest, (".mp4", ".mkv")) == expected


def test_manifest_rejects_missing_duplicate_and_unsupported_paths(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.touch()
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text(f"{video}\n{video}\n", encoding="utf-8")
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(["absent.mp4"]), encoding="utf-8")
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_text(str(tmp_path / "notes.csv"), encoding="utf-8")
    (tmp_path / "notes.csv").touch()

    with pytest.raises(ValueError, match="Duplicate video path"):
        load_video_manifest(duplicate, (".mp4",))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_video_manifest(missing, (".mp4",))
    with pytest.raises(ValueError, match="Unsupported video extension"):
        load_video_manifest(unsupported, (".mp4",))


def test_persistent_model_group_starts_all_gpus_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(4)

    def fake_start(_model, _python, _checkpoint, gpu_id, ready, **_kwargs):
        barrier.wait(timeout=2)
        replica_id = int(ready.stem.rsplit("_", 1)[-1])
        return SimpleNamespace(
            endpoint=DepthServiceEndpoint(
                "127.0.0.1", 20000 + gpu_id * 10 + replica_id, "00" * 32
            ),
            stop=lambda: None,
        )

    monkeypatch.setattr("camera_create.batch.start_depth_service", fake_start)
    options = BatchOptions(
        input_manifest=tmp_path / "clip.txt",
        checkpoint_root=tmp_path / "state",
        model_paths=ModelPaths(tmp_path / "pi3x", tmp_path / "moge3", tmp_path / "vipe"),
        pipeline_options=PipelineOptions(),
        gpu_ids=(0, 1, 2, 3),
        depth_services_per_gpu=2,
    )

    services, endpoints = _start_depth_service_group(
        "Pi3X", options.gpu_ids, options, tmp_path / "services"
    )

    assert len(services) == 8
    assert {key: endpoint.port for key, endpoint in endpoints.items()} == {
        (0, 0): 20000,
        (1, 0): 20010,
        (2, 0): 20020,
        (3, 0): 20030,
        (0, 1): 20001,
        (1, 1): 20011,
        (2, 1): 20021,
        (3, 1): 20031,
    }


def test_four_workers_bind_one_to_one_to_four_depth_replicas() -> None:
    assert [depth_service_replica_id(worker, 4, 4) for worker in range(8)] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


def test_duplicate_stems_in_one_directory_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "same.mp4"
    second = tmp_path / "same.mkv"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="same cam_<stem>.json"):
        validate_unique_camera_outputs([first, second])


def test_existing_camera_json_cannot_be_reused_by_another_extension(
    tmp_path: Path,
) -> None:
    video = tmp_path / "same.mkv"
    video.touch()
    camera_json_path(video).write_text(
        json.dumps({"format_version": 2, "video_name": "same.mp4"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belongs to same.mp4"):
        validate_existing_output_owners([video])


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
        json.dumps({"valid": True, "original_width": 1000, "original_height": 700}),
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.touch()
    output = camera_json_path(video)
    payload = export_camera_json_v2(
        result,
        output,
        video.name,
        30.0,
        24.0,
        241,
        10.06,
        (1920, 1080),
        (1000, 700),
    )
    assert payload["format_version"] == 2
    assert payload["is_metric"] is True
    assert payload["max_frames"] == 241
    assert payload["frames"][1]["timestamp_seconds"] == 1 / 24
    assert payload["frames"][1]["c2w"][0][3] == 1.25
    assert payload["frames"][0]["intrinsics"][0][0] == 500
    assert payload["frames"][0]["intrinsics_normalized"][0][0] == 0.5
    assert payload["frames"][0]["intrinsics_normalized"][1][1] == pytest.approx(0.7)
    assert payload["source_resolution"] == {"width": 1920, "height": 1080}
    assert payload["intrinsics_inference_resolution"] == {
        "width": 1000,
        "height": 700,
    }
    assert payload["metric_scale_validated_against_ground_truth"] is False
    assert valid_existing_output(video)
    assert valid_existing_output(video, max_frames=241)
    assert not valid_existing_output(video, vipe_height=480)
    assert valid_existing_output(video, vipe_height=700)
    assert not valid_existing_output(video, max_frames=120)
    assert not valid_existing_output(video, target_fps=30.0)
    assert not valid_existing_output(video, max_video_seconds=5.0)


def test_normalized_intrinsics_are_resolution_independent() -> None:
    first = np.array([[[1000.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0, 0, 1]]])
    second = first.copy()
    second[:, 0, :] *= 0.5
    second[:, 1, :] *= 0.5
    assert np.allclose(
        normalize_intrinsics_k(first, 1280, 720),
        normalize_intrinsics_k(second, 640, 360),
    )


def test_prepare_video_applies_fps_frame_and_duration_limits(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mkv"
    source.touch()
    commands: list[list[str]] = []

    def fake_probe(path: Path) -> tuple[float, int, float, int, int]:
        if path == source:
            return 30.0, 300, 10.0, 1920, 1080
        if ".vipe_720p" in path.name:
            return 24.0, 241, 241 / 24, 1280, 720
        return 24.0, 241, 241 / 24, 1920, 1080

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        Path(command[-1]).touch()

    monkeypatch.setattr("camera_create.batch._probe_video", fake_probe)
    monkeypatch.setattr("camera_create.batch.shutil.which", lambda _name: "/bin/ffmpeg")
    monkeypatch.setattr("camera_create.batch.subprocess.run", fake_run)

    (
        processed,
        vipe_video,
        source_fps,
        max_seconds,
        source_size,
        inference_size,
    ) = prepare_video(source, tmp_path / "work", 24.0, 241, 10.06, 720, "ffmpeg")

    assert processed.is_file()
    assert source_fps == 30.0
    assert max_seconds == 10.06
    assert vipe_video.is_file()
    assert source_size == (1920, 1080)
    assert inference_size == (1280, 720)
    assert commands[0][commands[0].index("-vf") + 1] == "fps=24.0"
    assert commands[1][commands[1].index("-vf") + 1] == "scale=-2:720"
    assert commands[0][commands[0].index("-frames:v") + 1] == "241"
    assert commands[0][commands[0].index("-t") + 1] == "10.06"
