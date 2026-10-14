"""End-to-end pipeline: download -> validate -> split -> tune -> evaluate -> save.

Run with::

    python -m recsys_lab.pipeline             # ./data and ./artifacts
    python -m recsys_lab.pipeline --no-tune   # skip the validation-tail search
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import paths
from .data import download_ml100k, load_dataset, validate_ml100k
from .evaluate import (
    DEFAULT_K_VALUES,
    PRIMARY_K,
    print_table,
    run_experiment,
    save_results,
)
from .models import model_families
from .split import three_way_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommender Systems Lab pipeline")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-test", type=int, default=20)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-tune", action="store_true", help="use default params")
    parser.add_argument("--models", default="all", help="comma-separated families, or 'all'")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    t_start = time.perf_counter()

    print(f"[1/6] MovieLens 100K -> {paths.RAW_DIR}")
    raw_dir = download_ml100k(paths.RAW_DIR, force=args.force_download)

    print("[2/6] Loading + validating against GroupLens published counts")
    data = load_dataset(raw_dir)
    stats = validate_ml100k(data)
    print("      " + json.dumps(stats))

    print("[3/6] Three-way temporal split")
    splits = three_way_split(
        data,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        max_test=args.max_test,
    )
    print("      " + json.dumps(splits.stats()))

    families = model_families()
    if args.models != "all":
        wanted = {m.strip() for m in args.models.split(",")}
        families = [f for f in families if f in wanted]
        if not families:
            print(f"      no models matched {sorted(wanted)}", file=sys.stderr)
            return 2

    if args.no_tune:
        print("[4/6] Skipping hyperparameter search (default params)")
    else:
        print("[4/6] Tuning hyperparameters on the validation tail")
    results = run_experiment(
        data, splits, k_values=DEFAULT_K_VALUES, families=families, tune=not args.no_tune
    )

    print("[5/6] Saving results")
    results["dataset"] = stats
    results["dataset"]["source"] = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    save_results(results, paths.RESULTS_JSON, paths.RESULTS_CSV)
    print(f"      {paths.RESULTS_JSON}")
    print(f"      {paths.RESULTS_CSV}")

    print(f"[6/6] Leaderboard on the held-out test tail @ K={PRIMARY_K}")
    print_table(results, PRIMARY_K)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s")
    return 0


def load_artifacts() -> dict[str, object]:
    """Load saved evaluation results (used by the dashboard)."""
    with paths.RESULTS_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    raise SystemExit(main())
