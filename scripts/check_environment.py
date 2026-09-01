#!/usr/bin/env python3
"""Check Python, CUDA, model checkpoints, Pi3X, MoGe-2, and VIPE before inference."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Print readiness checks and return nonzero if a required item is missing."""
    checks: list[tuple[str, bool, str]] = []
    checks.append(
        ("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    )
    for module in ("torch", "cv2", "scipy", "pi3", "moge"):
        checks.append(
            (f"import {module}", importlib.util.find_spec(module) is not None, "")
        )
    checks.append(
        (
            "VIPE CLI",
            shutil.which("vipe") is not None,
            shutil.which("vipe") or "not found",
        )
    )
    for model in ("pi3x", "moge2"):
        path = PROJECT_ROOT / "ckpt" / model
        populated = path.exists() and any(
            item.name != ".gitkeep" for item in path.iterdir()
        )
        checks.append((f"checkpoint {model}", populated, str(path)))
    try:
        import torch

        checks.append(
            (
                "CUDA available",
                bool(torch.cuda.is_available()),
                str(torch.cuda.is_available()),
            )
        )
    except ImportError:
        checks.append(("CUDA available", False, "torch missing"))
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'MISSING'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
