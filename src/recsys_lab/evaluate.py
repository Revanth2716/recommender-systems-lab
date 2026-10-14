"""Evaluation harness.

All models are scored under one protocol so the comparison table is meaningful:

1. Fit on the training interactions only.
2. Score every (user, item) pair.
3. Mask items the user already rated in training.
4. Take the top-K items.
5. Compute Precision@K, Recall@K and NDCG@K against the held-out items, then
   average across users.

``run_experiment`` first tunes each family on the validation tail, refits the
winner on train+validation, and only then scores it on the test tail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GENRES, MovieLensData
from .metrics import catalog_coverage, evaluate_ranking
from .models import Content, Recommender, build_content, make_model, model_families
from .ranking import top_k_matrix
from .split import Split, ThreeWay
from .tuning import TUNING_K, TUNING_METRIC, tune_all

DEFAULT_K_VALUES = (5, 10, 20)
PRIMARY_K = 10


def build_content_from(data: MovieLensData) -> Content:
    """Content features for the dataset's item metadata."""
    return build_content(data.items, GENRES)


def evaluate_model(
    model: Recommender,
    split: Split,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    *,
    train_seconds: float = 0.0,
) -> dict[str, object]:
    """Compute metrics at every K for one fitted model on a split's test items."""
    t0 = time.perf_counter()
    scores = model.score_all()
    score_seconds = time.perf_counter() - t0
    test_sets = split.test_sets()
    seen = model.seen_items()

    out: dict[str, object] = {
        "params": dict(getattr(model, "params", {})),
        "label": model.label,
        "kind": model.kind,
        "train_seconds": round(train_seconds, 4),
        "score_seconds": round(score_seconds, 4),
        "n_items": int(scores.shape[1]),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "metrics": {},
    }

    recs_max = top_k_matrix(scores, max(k_values), seen)
    coverage = catalog_coverage(recs_max) / scores.shape[1]

    for k in k_values:
        recs = {u: items[:k] for u, items in recs_max.items()}
        metrics = evaluate_ranking(recs, test_sets, k)
        metrics["coverage"] = coverage
        out["metrics"][str(k)] = {key: round(val, 6) for key, val in metrics.items()}

    out["sample_recommendations"] = {str(u): recs_max[u] for u in sorted(recs_max)[:3]}
    return out


def fit_and_evaluate(
    model: Recommender,
    split: Split,
    content: Content,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, object]:
    t0 = time.perf_counter()
    model.fit(split.train_matrix(), content)
    train_seconds = time.perf_counter() - t0
    return evaluate_model(model, split, k_values, train_seconds=train_seconds)


def run_experiment(
    data: MovieLensData,
    splits: ThreeWay,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    families: list[str] | None = None,
    tune: bool = True,
) -> dict[str, object]:
    """Tune on validation, refit on train+validation, report on the test tail."""
    content = build_content_from(data)
    families = families or model_families()

    tuning: dict[str, object] = {}
    results: dict[str, object] = {
        "k_values": list(k_values),
        "primary_k": PRIMARY_K,
        "tuning": tuning,
        "model_results": {},
        "protocol": {},
    }

    if tune:
        tuned = tune_all(families, splits.core, content, k=TUNING_K)
        for name, res in tuned.items():
            tuning[name] = {
                "metric": f"{TUNING_METRIC}@{TUNING_K}",
                "split": "validation tail (fit on core train period)",
                "best_params": res.best_params,
                "best_validation_score": round(res.best.score, 6),
                "seconds": res.seconds,
                "candidates": [{"params": c.params, "metrics": c.metrics} for c in res.candidates],
            }
        best_params = {name: tuning[name]["best_params"] for name in families}
    else:
        best_params = {name: {} for name in families}

    for name in families:
        model = make_model(name, best_params[name])
        results["model_results"][name] = fit_and_evaluate(model, splits.final, content, k_values)

    results["protocol"] = {
        "schema": "per-user temporal leave-last-n, all-unrated-item ranking",
        "split": splits.stats(),
        "k_values": list(k_values),
        "primary_k": PRIMARY_K,
        "tuning_metric": f"{TUNING_METRIC}@{TUNING_K} on the validation tail",
        "test_used_for_tuning": False,
    }
    results["comparison"] = comparison_table(results)
    return results


def comparison_table(results: dict[str, object]) -> list[dict[str, object]]:
    """Flatten model results into rows: one per (model, k)."""
    rows: list[dict[str, object]] = []
    for name, res in results["model_results"].items():  # type: ignore[union-attr]
        for k, m in res["metrics"].items():
            rows.append(
                {
                    "model": name,
                    "label": res["label"],
                    "kind": res["kind"],
                    "k": int(k),
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "ndcg": m["ndcg"],
                    "coverage": m["coverage"],
                    "train_seconds": res["train_seconds"],
                    "params": json.dumps(res["params"], sort_keys=True),
                }
            )
    return rows


def save_results(results: dict[str, object], json_path: Path, csv_path: Path) -> None:
    """Write the full result dict as JSON and the flat comparison as CSV.

    The JSON keeps the ``comparison`` rows as well as the nested per-model detail,
    so the dashboard can be driven from a single self-contained artifact.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    pd.DataFrame(results["comparison"]).to_csv(csv_path, index=False)


def print_table(results: dict[str, object], k: int = PRIMARY_K) -> None:
    """Print a readable leaderboard at K."""
    rows = [r for r in results["comparison"] if r["k"] == k]  # type: ignore[index]
    rows.sort(key=lambda r: -r["ndcg"])
    header = (
        f"{'model':<34}{'kind':<14}{'P@' + str(k):>8}"
        f"{'R@' + str(k):>8}{'NDCG@' + str(k):>9}{'cov':>7}{'fit s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<34}{r['kind']:<14}{r['precision']:>8.4f}"
            f"{r['recall']:>8.4f}{r['ndcg']:>9.4f}{r['coverage']:>7.3f}"
            f"{r['train_seconds']:>8.2f}"
        )
