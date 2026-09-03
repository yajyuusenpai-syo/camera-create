"""Manage atomic, fingerprinted resume metadata for expensive pipeline stages."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

CACHE_FORMAT_VERSION = 1
GENERATED_FILES = (
    "pi3x_depth.npz",
    "pi3x_depth.npz.tmp",
    "moge3_depth.npz",
    "moge3_depth.npz.tmp",
    "metric_depth_cache.npz",
    "metric_depth_cache.npz.tmp",
)


def video_identity(video: Path) -> dict[str, Any]:
    """Identify a local video without hashing all of its potentially large bytes."""
    stat = video.stat()
    return {
        "path": str(video.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def fingerprint(payload: dict[str, Any]) -> str:
    """Create a deterministic cache key from JSON-compatible pipeline inputs."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class StageCache:
    """Track completed stages and invalidate stale generated artifacts safely."""

    def __init__(self, root: Path, key: str, context: dict[str, Any]):
        self.root = root.resolve()
        self.manifest = self.root / "stage_state.json"
        self.key = key
        self.context = context

    def prepare(self) -> bool:
        """Return True for a matching prior run, otherwise initialize fresh state."""
        self.root.mkdir(parents=True, exist_ok=True)
        previous: dict[str, Any] = {}
        if self.manifest.is_file():
            try:
                previous = json.loads(self.manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        reusable = bool(
            previous.get("format_version") == CACHE_FORMAT_VERSION
            and previous.get("fingerprint") == self.key
        )
        if reusable:
            return True
        for filename in GENERATED_FILES:
            (self.root / filename).unlink(missing_ok=True)
        vipe_dir = self.root / "vipe"
        if vipe_dir.is_dir():
            shutil.rmtree(vipe_dir)
        self._write({"completed_stages": []})
        return False

    def completed(self, stage: str) -> None:
        """Atomically record one completed stage for diagnostics and batch checkpoints."""
        state = self.read()
        stages = list(state.get("completed_stages", []))
        if stage not in stages:
            stages.append(stage)
        self._write({"completed_stages": stages})

    def read(self) -> dict[str, Any]:
        """Read current state, returning an empty dictionary if it is unavailable."""
        try:
            return json.loads(self.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, changes: dict[str, Any]) -> None:
        state = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": self.key,
            "context": self.context,
            **changes,
        }
        temporary = self.manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.manifest)
