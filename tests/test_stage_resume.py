"""Test fingerprint invalidation and stage-level pipeline resume without loading models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera_create.config import ModelPaths
from camera_create.pipeline import CameraCreatePipeline, PipelineOptions
from camera_create.stage_cache import StageCache, fingerprint
from camera_create.worker_runner import DepthWorkerResult


def _worker_result() -> DepthWorkerResult:
    return DepthWorkerResult(
        depth=np.ones((2, 8, 8), dtype=np.float32),
        frame_count=2,
        original_width=16,
        original_height=16,
        inference_width=8,
        inference_height=8,
        fps=24.0,
    )


def _write_worker(path: Path, result: DepthWorkerResult) -> None:
    np.savez_compressed(
        path,
        depth=result.depth,
        frame_count=result.frame_count,
        original_width=result.original_width,
        original_height=result.original_height,
        inference_width=result.inference_width,
        inference_height=result.inference_height,
        fps=result.fps,
    )


def test_stage_cache_invalidates_generated_files(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    first = StageCache(root, fingerprint({"input": 1}), {"input": 1})
    assert not first.prepare()
    (root / "pi3x_depth.npz").write_bytes(b"old")
    first.completed("pi3x")
    assert StageCache(root, first.key, {"input": 1}).prepare()

    changed = StageCache(root, fingerprint({"input": 2}), {"input": 2})
    assert not changed.prepare()
    assert not (root / "pi3x_depth.npz").exists()


def test_pipeline_reuses_all_completed_stages(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    pi3x_ckpt = tmp_path / "pi3x"
    moge_ckpt = tmp_path / "moge3"
    pi3x_ckpt.mkdir()
    moge_ckpt.mkdir()
    (pi3x_ckpt / "model").touch()
    (moge_ckpt / "model").touch()
    models = ModelPaths(pi3x_ckpt, moge_ckpt, tmp_path / "vipe-cache")
    calls = {"pi3x": 0, "moge3": 0, "vipe": 0}

    def fake_pi3x(*args):
        calls["pi3x"] += 1
        result = _worker_result()
        _write_worker(args[3], result)
        return result

    def fake_moge3(*args):
        calls["moge3"] += 1
        base = _worker_result()
        result = DepthWorkerResult(
            depth=base.depth * 3,
            frame_count=base.frame_count,
            original_width=base.original_width,
            original_height=base.original_height,
            inference_width=base.inference_width,
            inference_height=base.inference_height,
            fps=base.fps,
        )
        _write_worker(args[3], result)
        return result

    def fake_vipe(_video, output, *_args):
        calls["vipe"] += 1
        if calls["vipe"] == 1:
            raise RuntimeError("simulated VIPE failure")
        output.mkdir(parents=True)

    monkeypatch.setattr("camera_create.pipeline.run_pi3x_worker", fake_pi3x)
    monkeypatch.setattr("camera_create.pipeline.run_moge3_worker", fake_moge3)
    monkeypatch.setattr("camera_create.pipeline.run_vipe", fake_vipe)
    monkeypatch.setattr(
        "camera_create.pipeline.preflight_vipe_integration", lambda *_args: None
    )
    monkeypatch.setattr(
        "camera_create.pipeline.export_camera_artifacts", lambda *_args: {"valid": True}
    )
    options = PipelineOptions(allow_vipe_downloads=True)
    pipeline = CameraCreatePipeline(models, options)
    work = tmp_path / "resume"
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="simulated VIPE failure"):
        pipeline.run(video, output, work)
    pipeline.run(video, output, work)
    pipeline.run(video, output, work)

    assert calls == {"pi3x": 1, "moge3": 1, "vipe": 2}
    assert (work / "metric_depth_cache.npz").is_file()
