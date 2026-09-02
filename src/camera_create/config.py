"""Resolve project paths and validate checkpoint locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class ModelPaths:
    """Filesystem locations for the three model components."""

    pi3x: Path
    moge3: Path
    vipe: Path

    @classmethod
    def defaults(cls, ckpt_root: Path | None = None) -> ModelPaths:
        root = (ckpt_root or PROJECT_ROOT / "ckpt").resolve()
        return cls(root / "pi3x", root / "moge3", root / "vipe")

    def validate_depth_models(self) -> None:
        missing = []
        for path in (self.pi3x, self.moge3):
            populated = path.is_file() or (
                path.is_dir()
                and any(item.name != ".gitkeep" for item in path.iterdir())
            )
            if not populated:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint path(s): "
                + ", ".join(missing)
                + ". See camera_create/docs/DEPLOYMENT.md."
            )
