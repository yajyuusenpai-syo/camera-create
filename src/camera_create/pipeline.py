"""Orchestrate isolated Pi3X + MoGe-3 + VIPE metric camera inference."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import export_camera_artifacts
from .config import ModelPaths
from .depth import default_fov_x, fuse_metric_depths, save_depth_cache
from .vipe_runner import preflight_vipe_assets, run_vipe
from .worker_runner import (
    default_environment_executable,
    ensure_matching_workers,
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
        owned_work = work_dir is None
        actual_work = (
            work_dir.resolve()
            if work_dir
            else Path(tempfile.mkdtemp(prefix="camera-create-"))
        )
        actual_work.mkdir(parents=True, exist_ok=True)
        try:
            pi3x_cache = actual_work / "pi3x_depth.npz"
            moge3_cache = actual_work / "moge3_depth.npz"
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
            fov_x = self.options.fov_x_deg or default_fov_x(
                pi3x_result.original_width
            )
            LOG.info("Running isolated MoGe-3 worker: %s", self.options.moge3_python)
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
            ensure_matching_workers(pi3x_result, moge3_result)
            fused, scale, raw_scale = fuse_metric_depths(
                pi3x_result.depth, moge3_result.depth, self.options.ema_momentum
            )
            cache_path = actual_work / "metric_depth_cache.npz"
            save_depth_cache(cache_path, fused, scale, raw_scale)
            vipe_dir = actual_work / "vipe"
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
