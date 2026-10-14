"""Shared pytest fixtures.

The real MovieLens 100K dataset is downloaded on first use so the suite tests the
actual data contract, not a synthetic stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recsys_lab import paths  # noqa: E402
from recsys_lab.data import MovieLensData, download_ml100k, load_dataset  # noqa: E402
from recsys_lab.evaluate import build_content_from  # noqa: E402
from recsys_lab.models import Content  # noqa: E402
from recsys_lab.split import ThreeWay, three_way_split  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    paths.ensure_dirs()
    if not paths.ML100K_DIR.joinpath("u.data").exists():
        download_ml100k(paths.RAW_DIR)
    return paths.ML100K_DIR


@pytest.fixture(scope="session")
def data(raw_dir: Path) -> MovieLensData:
    return load_dataset(raw_dir)


@pytest.fixture(scope="session")
def splits(data: MovieLensData) -> ThreeWay:
    return three_way_split(data)


@pytest.fixture(scope="session")
def content(data: MovieLensData) -> Content:
    return build_content_from(data)
