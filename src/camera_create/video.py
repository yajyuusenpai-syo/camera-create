"""Decode a video to uniformly resized RGB frames for depth inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoData:
    frames_rgb: np.ndarray
    original_width: int
    original_height: int
    fps: float

    @property
    def frame_count(self) -> int:
        return int(self.frames_rgb.shape[0])


def _inference_size(
    width: int, height: int, max_side: int, multiple: int = 14
) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    new_w = max(multiple, int(width * scale) // multiple * multiple)
    new_h = max(multiple, int(height * scale) // multiple * multiple)
    return new_w, new_h


def read_video(video_path: Path, max_inference_side: int = 560) -> VideoData:
    """Read RGB frames, resizing only the depth-inference copy to bound memory."""
    video_path = video_path.resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    infer_w, infer_h = _inference_size(width, height, max_inference_side)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (infer_w, infer_h) != (width, height):
            rgb = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return VideoData(np.stack(frames), width, height, fps)
