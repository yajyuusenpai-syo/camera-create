#!/usr/bin/env python3
"""Verify camera-create inference optimizations without loading models or running the CLI."""

from __future__ import annotations

import argparse
import ast
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    """Read one required project source file."""
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def function_calls(source: str, function_name: str) -> int:
    """Count direct calls to a named function using Python's syntax tree."""
    tree = ast.parse(source)
    return sum(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == function_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        )
        for node in ast.walk(tree)
    )


def result(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """Create one stable machine-readable check result."""
    return {"name": name, "passed": passed, "detail": detail}


def check_uncompressed_npz() -> dict[str, Any]:
    """Write a tiny NPZ exactly as optimized workers do and inspect ZIP members."""
    try:
        import numpy as np
    except ImportError:
        return result(
            "uncompressed_npz_runtime",
            False,
            "NumPy is unavailable; run this script with .envs/pi3x/bin/python",
        )
    with tempfile.TemporaryDirectory(prefix="camera-create-check-") as directory:
        path = Path(directory) / "probe.npz"
        np.savez(path, probe=np.arange(8, dtype=np.float32))
        with zipfile.ZipFile(path) as archive:
            stored = all(
                member.compress_type == zipfile.ZIP_STORED
                for member in archive.infolist()
            )
    return result(
        "uncompressed_npz_runtime",
        stored,
        "np.savez produced ZIP_STORED members" if stored else "NPZ members were compressed",
    )


def run_checks() -> list[dict[str, Any]]:
    """Inspect all optimized execution paths without starting any inference process."""
    batch = read("src/camera_create/batch.py")
    pipeline = read("src/camera_create/pipeline.py")
    vipe_worker_path = PROJECT_ROOT / "scripts/run_vipe_worker.py"
    vipe_worker = vipe_worker_path.read_text(encoding="utf-8") if vipe_worker_path.is_file() else ""
    pi3_worker = read("scripts/run_pi3x_worker.py")
    moge_worker = read("scripts/run_moge3_worker.py")
    depth = read("src/camera_create/depth.py")
    cli = read("src/camera_create/cli.py")

    checks = [
        result(
            "machine_level_vipe_preflight",
            function_calls(batch, "preflight_vipe_integration") == 1
            and "preflight_done=True" in batch
            and "if not self.options.preflight_done" in pipeline,
            "one controller preflight; child pipelines skip it",
        ),
        result(
            "local_tmp_intermediates",
            'Path(tempfile.gettempdir()).resolve() / "camera-create"' in batch
            and 'job_root = local_run_root / "stage_cache"' in batch
            and "--local-work-root" in cli,
            "default local work root is <system tmp>/camera-create",
        ),
        result(
            "uncompressed_npz_sources",
            all("np.savez(" in source and "np.savez_compressed(" not in source
                for source in (pi3_worker, moge_worker, depth)),
            "Pi3X, MoGe-3 and fused-depth writers use np.savez",
        ),
        check_uncompressed_npz(),
        result(
            "persistent_vipe_service",
            vipe_worker_path.is_file()
            and "start_vipe_service," in batch
            and "vipe_service=vipe_service" in batch
            and "pipeline.model_cache" not in vipe_worker
            and 'models.pop("depth/cached", None)' in vipe_worker,
            "VIPE service exists and resets only the video-bound cached depth",
        ),
        result(
            "fresh_slam_per_video",
            "pipeline.run(stream)" in vipe_worker
            and "make_vipe_pipeline" in vipe_worker
            and "--vipe-recycle-every" in cli,
            "one persistent pipeline handles new streams; periodic recycle is configurable",
        ),
        result(
            "single_decode_dual_output",
            "split=2[depth_src][vipe_src]" in batch
            and '"-filter_complex", filter_graph' in batch
            and 'str(processed),' in batch
            and 'str(vipe_video),' in batch,
            "one FFmpeg input/filter graph emits depth and VIPE videos",
        ),
        result(
            "flash_sdp_default_policy",
            'default=_environment_bool("CAMERA_CREATE_DISABLE_SDP")' in cli,
            "Flash SDP depends on --disable-sdp/CAMERA_CREATE_DISABLE_SDP at runtime",
        ),
    ]
    # Parse every changed Python entry point as a final zero-side-effect syntax check.
    for relative in (
        "src/camera_create/batch.py",
        "src/camera_create/pipeline.py",
        "src/camera_create/vipe_runner.py",
        "scripts/run_vipe_worker.py",
    ):
        ast.parse(read(relative), filename=relative)
    return checks


def main() -> int:
    """Print a concise report and return nonzero when any optimization is absent."""
    parser = argparse.ArgumentParser(
        description="Check optimization wiring without running camera inference"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()
    checks = run_checks()
    ready = all(item["passed"] for item in checks)
    report = {
        "project_root": str(PROJECT_ROOT),
        "inference_started": False,
        "ready": ready,
        "checks": checks,
        "note": (
            "This verifies code paths and defaults only; benchmark logs are still "
            "required to measure actual speedup."
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for item in checks:
            print(f"[{'OK' if item['passed'] else 'FAILED'}] {item['name']}: {item['detail']}")
        print(json.dumps({"ready": ready, "inference_started": False}))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
