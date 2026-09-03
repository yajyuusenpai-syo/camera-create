"""Test idempotent source patches required by the VIPE v1.2 cached-depth backend."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "setup_vipe.py"
SPEC = importlib.util.spec_from_file_location("setup_vipe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SETUP_VIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP_VIPE)
patch_depth_frame_index = SETUP_VIPE.patch_depth_frame_index


def test_patch_depth_frame_index_is_idempotent(tmp_path: Path) -> None:
    base = tmp_path / "base.py"
    buffer = tmp_path / "buffer.py"
    base.write_text(
        "class DepthEstimationInput:\n"
        "    camera_type: CameraType = CameraType.PINHOLE\n",
        encoding="utf-8",
    )
    buffer.write_text(
        "            depth_input = DepthEstimationInput(\n"
        "                camera_type=self.camera_type,\n"
        "            )\n",
        encoding="utf-8",
    )

    patch_depth_frame_index(base, buffer)
    first_base = base.read_text(encoding="utf-8")
    first_buffer = buffer.read_text(encoding="utf-8")
    patch_depth_frame_index(base, buffer)

    assert base.read_text(encoding="utf-8") == first_base
    assert buffer.read_text(encoding="utf-8") == first_buffer
    assert first_base.count("frame_idx: int | None = None") == 1
    assert first_buffer.count("frame_idx=int(self.tstamp[frame_idx].item())") == 1
