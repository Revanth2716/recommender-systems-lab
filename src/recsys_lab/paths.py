"""Project paths. Everything stays on the drive the repo lives on (D:)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_data_override = os.environ.get("RECSYS_DATA_DIR")
_artifacts_override = os.environ.get("RECSYS_ARTIFACTS_DIR")

DATA_DIR = Path(_data_override) if _data_override else PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ML100K_DIR = RAW_DIR / "ml-100k"
ARTIFACTS_DIR = Path(_artifacts_override) if _artifacts_override else PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

RESULTS_JSON = METRICS_DIR / "evaluation.json"
RESULTS_CSV = METRICS_DIR / "comparison.csv"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
