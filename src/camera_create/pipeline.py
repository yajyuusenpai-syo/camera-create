"""Orchestrate the complete Pi3X + MoGe-2 + VIPE metric camera pipeline."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import export_camera_artifacts
from .config import ModelPaths
from .depth import (
    default_fov_x,
    fuse_metric_depths,
    infer_moge2,
    infer_pi3x,
    load_moge2,
    load_pi3x,
    save_depth_cache,
)
from .video import read_video
from .vipe_runner import run_vipe

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
    vipe_command: str = "vipe"


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
        owned_work = work_dir is None
        actual_work = (
            work_dir.resolve()
            if work_dir
            else Path(tempfile.mkdtemp(prefix="camera-create-"))
        )
        actual_work.mkdir(parents=True, exist_ok=True)
        try:
            LOG.info("Decoding inference frames from %s", video)
            video_data = read_video(video, self.options.max_inference_side)
            fov_x = self.options.fov_x_deg or default_fov_x(video_data.original_width)
            LOG.info("Loading Pi3X from %s", self.models.pi3x)
            pi3x = load_pi3x(self.models.pi3x, self.options.device)
            pi3x_depth = infer_pi3x(
                pi3x,
                video_data.frames_rgb,
                self.options.device,
                self.options.pi3x_chunk,
                self.options.pi3x_stride,
            )
            del pi3x
            LOG.info("Loading MoGe-2 from %s", self.models.moge2)
            moge2 = load_moge2(self.models.moge2, self.options.device)
            moge2_depth = infer_moge2(
                moge2, video_data.frames_rgb, self.options.device, fov_x
            )
            del moge2
            fused, scale, raw_scale = fuse_metric_depths(
                pi3x_depth, moge2_depth, self.options.ema_momentum
            )
            cache_path = actual_work / "metric_depth_cache.npz"
            save_depth_cache(cache_path, fused, scale, raw_scale)
            vipe_dir = actual_work / "vipe"
            LOG.info("Running VIPE metric bundle adjustment")
            run_vipe(
                video, vipe_dir, cache_path, self.models.vipe, self.options.vipe_command
            )
            metadata = {
                "input_video": str(video),
                "frame_count": video_data.frame_count,
                "original_width": video_data.original_width,
                "original_height": video_data.original_height,
                "fps": video_data.fps,
                "depth_inference_width": int(video_data.frames_rgb.shape[2]),
                "depth_inference_height": int(video_data.frames_rgb.shape[1]),
                "fov_x_deg_for_moge2": fov_x,
                "translation_unit": "metre",
                "pose_convention": "OpenCV c2w and w2c; +x right, +y down, +z forward",
                "metric_basis": "MoGe-2 metric depth fused into Pi3X temporal depth and injected into VIPE BA",
            }
            report = export_camera_artifacts(
                video, vipe_dir, output_dir, video_data.frame_count, scale, metadata
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
