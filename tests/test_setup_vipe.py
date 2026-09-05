"""Test idempotent source patches required by the VIPE v1.2 cached-depth backend."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "setup_vipe.py"
SPEC = importlib.util.spec_from_file_location("setup_vipe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SETUP_VIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP_VIPE)
patch_depth_frame_index = SETUP_VIPE.patch_depth_frame_index
prepare_eigen_headers = SETUP_VIPE.prepare_eigen_headers


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


def test_prepare_eigen_headers_copies_offline_source(tmp_path: Path) -> None:
    vipe_source = tmp_path / "vipe"
    eigen_source = tmp_path / "headers" / "eigen3" / "Eigen"
    eigen_source.mkdir(parents=True)
    (eigen_source / "Core").write_text("// Eigen test header\n", encoding="utf-8")

    target = prepare_eigen_headers(vipe_source, eigen_source.parent)

    assert target == vipe_source / "csrc" / "include" / "eigen3" / "Eigen"
    assert (target / "Core").read_text(encoding="utf-8") == "// Eigen test header\n"


def test_prepare_eigen_headers_rejects_implicit_download(tmp_path: Path) -> None:
    try:
        prepare_eigen_headers(tmp_path / "vipe", tmp_path / "missing")
    except RuntimeError as error:
        assert "--eigen-source" in str(error)
    else:
        raise AssertionError("missing Eigen headers should fail before setup.py")
