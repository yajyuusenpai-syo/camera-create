"""Test detection used to avoid redundant editable package installations."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "editable_install_status.py"
SPEC = importlib.util.spec_from_file_location("editable_install_status_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


class FakeDistribution:
    """Expose minimal PEP 610 metadata for an installed distribution."""

    def __init__(self, direct_url: dict[str, object] | None) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json" or self.direct_url is None:
            return None
        return json.dumps(self.direct_url)


def test_detects_matching_editable_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    distribution = FakeDistribution(
        {"url": source.as_uri(), "dir_info": {"editable": True}}
    )
    monkeypatch.setattr(STATUS, "distributions", lambda: [distribution])

    assert STATUS.is_editable_source_installed(source)


def test_rejects_non_editable_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    distribution = FakeDistribution(
        {"url": source.as_uri(), "dir_info": {"editable": False}}
    )
    monkeypatch.setattr(STATUS, "distributions", lambda: [distribution])

    assert not STATUS.is_editable_source_installed(source)
