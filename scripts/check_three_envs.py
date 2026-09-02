#!/usr/bin/env python3
"""Probe the isolated Pi3X, MoGe-3, and VIPE interpreters without importing them together."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_probe(python: Path, source: str) -> tuple[bool, str]:
    """Run a JSON-emitting probe in one isolated interpreter."""
    if not python.is_file():
        return False, f"interpreter missing: {python}"
    result = subprocess.run(
        [str(python), "-c", source], capture_output=True, text=True, check=False
    )
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0, detail


def main() -> int:
    """Check imports, versions, CUDA visibility, and checkpoint directories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-root", type=Path, default=Path(".envs"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Only validate environments; useful before weights are downloaded.",
    )
    args = parser.parse_args()
    env_root = args.env_root.resolve()
    project_root = args.project_root.resolve()
    probes = {
        "pi3x": "import json,numpy,torch; from pi3 import Pi3X; "
        "print(json.dumps({'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.cuda.is_available()}))",
        "moge3": "import json,numpy,torch; from moge.model.v3 import MoGeModel; "
        "print(json.dumps({'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.cuda.is_available()}))",
        "vipe": "import json,torch,vipe; "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available()}))",
    }
    all_ok = True
    for name, source in probes.items():
        python = env_root / name / "bin" / "python"
        ok, detail = run_probe(python, source)
        all_ok &= ok
        print(f"[{'OK' if ok else 'FAILED'}] {name}: {detail}")
    for name in (() if args.skip_checkpoints else ("pi3x", "moge3", "vipe")):
        checkpoint = project_root / "ckpt" / name
        populated = checkpoint.is_dir() and any(
            item.name != ".gitkeep" for item in checkpoint.iterdir()
        )
        print(f"[{'OK' if populated else 'MISSING'}] checkpoint {name}: {checkpoint}")
        all_ok &= populated
    print(json.dumps({"ready": all_ok, "env_root": str(env_root)}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
