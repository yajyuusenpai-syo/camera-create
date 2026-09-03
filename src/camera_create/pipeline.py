"""Orchestrate isolated Pi3X + MoGe-3 + VIPE metric camera inference."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import export_camera_artifacts
from .config import ModelPaths
from .depth import (
    default_fov_x,
    fuse_metric_depths,
    load_depth_cache,
    save_depth_cache,
)
from .stage_cache import StageCache, fingerprint, video_identity
from .vipe_runner import (
    preflight_vipe_assets,
    preflight_vipe_integration,
    run_vipe,
)
from .worker_runner import (
    default_environment_executable,
    ensure_matching_workers,
    load_worker_cache,
    run_moge3_worker,
    run_pi3x_worker,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    """Runtime and memory controls for the end-to-end pipeline."""

    device: str = "cuda"
    pi3x_chunk: int = 16
    pi3x_stride: int = 8
    ema_momentum: float = 0.99
    max_inference_side: int = 560
    fov_x_deg: float | None = None
    keep_work: bool = False
    pi3x_python: Path = field(
        default_factory=lambda: default_environment_executable("pi3x")
    )
    moge3_python: Path = field(
        default_factory=lambda: default_environment_executable("moge3")
    )
    vipe_command: str = field(
        default_factory=lambda: str(default_environment_executable("vipe", "vipe"))
    )
    moge3_refine_steps: int = 3
    moge3_fp16: bool = True
    allow_vipe_downloads: bool = False


class CameraCreatePipeline:
    """Reusable Python API behind the command-line interface."""

    def __init__(self, models: ModelPaths, options: PipelineOptions | None = None):
        self.models = models
        self.options = options or PipelineOptions()

    def run(self, video: Path, output_dir: Path, work_dir: Path | None = None) -> dict:
        """Process one video and return its validation/metric summary."""
        video = video.resolve()
        output_dir = output_dir.resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video}")
        self.models.validate_depth_models()
        preflight_vipe_assets(
            self.models.vipe, self.options.allow_vipe_downloads
        )
        preflight_vipe_integration(self.options.vipe_command)
        owned_work = work_dir is None
        actual_work = (
            work_dir.resolve()
            if work_dir
            else Path(tempfile.mkdtemp(prefix="camera-create-"))
        )
        actual_work.mkdir(parents=True, exist_ok=True)
        cache_context = {
            "video": video_identity(video),
            "pi3x_checkpoint": str(self.models.pi3x.resolve()),
            "moge3_checkpoint": str(self.models.moge3.resolve()),
            "pi3x_chunk": self.options.pi3x_chunk,
            "pi3x_stride": self.options.pi3x_stride,
            "ema_momentum": self.options.ema_momentum,
            "max_inference_side": self.options.max_inference_side,
            "fov_x_deg": self.options.fov_x_deg,
            "moge3_refine_steps": self.options.moge3_refine_steps,
            "moge3_fp16": self.options.moge3_fp16,
            "pi3x_python": str(self.options.pi3x_python),
            "moge3_python": str(self.options.moge3_python),
        }
        stage_cache = StageCache(
            actual_work, fingerprint(cache_context), cache_context
        )
        resumed = stage_cache.prepare()
        if resumed:
            LOG.info("Found matching stage-resume cache: %s", actual_work)
        try:
            pi3x_cache = actual_work / "pi3x_depth.npz"
            moge3_cache = actual_work / "moge3_depth.npz"
            try:
                pi3x_result = load_worker_cache(pi3x_cache, "Pi3X")
                LOG.info("[resume] Reusing Pi3X depth: %s", pi3x_cache)
                stage_cache.completed("pi3x")
            except Exception:  # noqa: BLE001 - any corrupt/incomplete cache is rebuilt
                pi3x_cache.unlink(missing_ok=True)
                LOG.info("Running isolated Pi3X worker: %s", self.options.pi3x_python)
                pi3x_result = run_pi3x_worker(
                    self.options.pi3x_python,
                    video,
                    self.models.pi3x,
                    pi3x_cache,
                    self.options.device,
                    self.options.pi3x_chunk,
                    self.options.pi3x_stride,
                    self.options.max_inference_side,
                )
                stage_cache.completed("pi3x")
            fov_x = self.options.fov_x_deg or default_fov_x(
                pi3x_result.original_width
            )
            try:
                moge3_result = load_worker_cache(moge3_cache, "MoGe-3")
                LOG.info("[resume] Reusing MoGe-3 depth: %s", moge3_cache)
                stage_cache.completed("moge3")
            except Exception:  # noqa: BLE001 - any corrupt/incomplete cache is rebuilt
                moge3_cache.unlink(missing_ok=True)
                LOG.info(
                    "Running isolated MoGe-3 worker: %s", self.options.moge3_python
                )
                moge3_result = run_moge3_worker(
                    self.options.moge3_python,
                    video,
                    self.models.moge3,
                    moge3_cache,
                    self.options.device,
                    self.options.max_inference_side,
                    fov_x,
                    self.options.moge3_refine_steps,
                    self.options.moge3_fp16,
                )
                stage_cache.completed("moge3")
            ensure_matching_workers(pi3x_result, moge3_result)
            cache_path = actual_work / "metric_depth_cache.npz"
            try:
                _, scale, _ = load_depth_cache(
                    cache_path,
                    pi3x_result.frame_count,
                    (pi3x_result.inference_height, pi3x_result.inference_width),
                )
                LOG.info("[resume] Reusing fused metric depth: %s", cache_path)
                stage_cache.completed("metric_depth")
            except Exception:  # noqa: BLE001 - any corrupt/incomplete cache is rebuilt
                cache_path.unlink(missing_ok=True)
                fused, scale, raw_scale = fuse_metric_depths(
                    pi3x_result.depth,
                    moge3_result.depth,
                    self.options.ema_momentum,
                )
                save_depth_cache(cache_path, fused, scale, raw_scale)
                stage_cache.completed("metric_depth")
            vipe_dir = actual_work / "vipe"
            completed_stages = stage_cache.read().get("completed_stages", [])
            if "vipe" in completed_stages and vipe_dir.is_dir():
                LOG.info("[resume] Reusing completed VIPE output: %s", vipe_dir)
            else:
                if vipe_dir.is_dir():
                    shutil.rmtree(vipe_dir)
                LOG.info("Running VIPE metric bundle adjustment")
                run_vipe(
                    video,
                    vipe_dir,
                    cache_path,
                    self.models.vipe,
                    self.options.vipe_command,
                    self.options.allow_vipe_downloads,
                )
            metadata = {
                "input_video": str(video),
                "frame_count": pi3x_result.frame_count,
                "original_width": pi3x_result.original_width,
                "original_height": pi3x_result.original_height,
                "fps": pi3x_result.fps,
                "depth_inference_width": pi3x_result.inference_width,
                "depth_inference_height": pi3x_result.inference_height,
                "fov_x_deg_for_moge3": fov_x,
                "moge3_refine_steps": self.options.moge3_refine_steps,
                "translation_unit": "metre",
                "pose_convention": "OpenCV c2w and w2c; +x right, +y down, +z forward",
                "metric_basis": "MoGe-3 metric depth fused into Pi3X temporal depth and injected into VIPE BA",
            }
            report = export_camera_artifacts(
                video, vipe_dir, output_dir, pi3x_result.frame_count, scale, metadata
            )
            stage_cache.completed("vipe")
            if self.options.keep_work:
                retained = output_dir / "work"
                if retained.exists():
                    raise FileExistsError(
                        f"Cannot retain work; destination already exists: {retained}"
                    )
                shutil.copytree(actual_work, retained)
            return report
        finally:
            if owned_work:
                shutil.rmtree(actual_work, ignore_errors=True)
