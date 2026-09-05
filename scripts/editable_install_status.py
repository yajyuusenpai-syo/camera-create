#!/usr/bin/env python3
"""Check whether this Python environment already links an editable source tree."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


def file_url_path(url: str) -> Path | None:
    """Convert a local file URL from PEP 610 metadata into a filesystem path."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"
    return Path(path).resolve()


def is_editable_source_installed(source: Path) -> bool:
    """Return whether any installed editable distribution points at source."""
    expected = source.resolve()
    for distribution in distributions():
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            continue
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            continue
        if not direct_url.get("dir_info", {}).get("editable", False):
            continue
        installed_source = file_url_path(direct_url.get("url", ""))
        if installed_source == expected:
            return True
    return False


def main() -> int:
    """Exit successfully only when the requested source is already editable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    return 0 if is_editable_source_installed(args.source) else 1


if __name__ == "__main__":
    raise SystemExit(main())
