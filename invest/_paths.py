from __future__ import annotations

import os
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"

def package_root() -> Path:
    """Return the root directory of the installed invest package."""
    return PACKAGE_ROOT

def data_path(*relative: str) -> Path:
    """Return an absolute path inside the bundled data directory."""
    path = DATA_DIR.joinpath(*relative)
    print(path)
    return path