#!/usr/bin/env python3
"""Install the repository's VIPE checkout and idempotently register cached metric depth."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def patch_factory(factory_path: Path) -> None:
    """Add the cached backend branch to VIPE's model factory when absent."""
    source = factory_path.read_text(encoding="utf-8")
    if 'model_name == "cached"' in source:
        return
    marker = '    else:\n        raise ValueError(f"Unknown depth model: {model}")'
    insertion = (
        '    elif model_name == "cached":\n'
        "        import os\n"
        "        from .cached import CachedDepthModel\n"
        '        cache_path = os.environ.get("SANA_WM_CACHED_DEPTH_PATH", "")\n'
        "        if not cache_path:\n"
        '            raise ValueError("cached depth requires SANA_WM_CACHED_DEPTH_PATH")\n'
        "        return CachedDepthModel(cache_path)\n\n"
    )
    if marker not in source:
        raise RuntimeError(
            f"Cannot safely patch unfamiliar VIPE factory: {factory_path}"
        )
    factory_path.write_text(
        source.replace(marker, insertion + marker), encoding="utf-8"
    )


def main() -> int:
    """Install VIPE into the active Python environment and apply local files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vipe-source", type=Path, default=REPOSITORY_ROOT / "third_party" / "vipe"
    )
    parser.add_argument(
        "--constraint",
        type=Path,
        help="Optional pip constraint file used while installing VIPE.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Do not resolve dependencies; useful when repairing an existing environment.",
    )
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    vipe_source = args.vipe_source.resolve()
    if not (vipe_source / "pyproject.toml").exists():
        raise FileNotFoundError(f"VIPE source checkout not found: {vipe_source}")
    depth_dir = vipe_source / "vipe" / "priors" / "depth"
    config_dir = vipe_source / "configs" / "pipeline"
    shutil.copy2(PROJECT_ROOT / "patches" / "vipe_cached.py", depth_dir / "cached.py")
    shutil.copy2(
        PROJECT_ROOT / "configs" / "vipe_cached_depth.yaml",
        config_dir / "vipe_cached_depth.yaml",
    )
    patch_factory(depth_dir / "__init__.py")
    if not args.skip_install:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
        ]
        if args.constraint:
            command.extend(["--constraint", str(args.constraint.resolve())])
        if args.no_deps:
            command.append("--no-deps")
        command.extend(["-e", str(vipe_source)])
        subprocess.run(command, check=True)
    print(f"VIPE cached-depth integration ready at {vipe_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
