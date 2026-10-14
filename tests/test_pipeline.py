"""End-to-end smoke tests for the evaluation harness and artifact writing."""

from __future__ import annotations

import json

import pytest

from recsys_lab.evaluate import comparison_table, run_experiment, save_results
from recsys_lab.models import PARAM_GRIDS, model_families
from recsys_lab.tuning import tune_family


@pytest.fixture(scope="session")
def tiny_experiment(data, splits):
    return run_experiment(
        data,
        splits,
        k_values=(5, 10),
        families=["random", "popularity_count", "content_based"],
        tune=False,
    )


def test_experiment_reports_every_model_at_every_k(tiny_experiment):
    rows = tiny_experiment["comparison"]
    assert {r["k"] for r in rows} == {5, 10}
    assert {r["model"] for r in rows} == {"random", "popularity_count", "content_based"}
    for row in rows:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["ndcg"] <= 1.0
        assert 0.0 <= row["coverage"] <= 1.0


def test_experiment_records_the_protocol(tiny_experiment):
    protocol = tiny_experiment["protocol"]
    assert protocol["test_used_for_tuning"] is False
    assert "temporal leave-last-n" in protocol["schema"]
    assert protocol["split"]["final"]["n_test"] > 0


def test_save_results_writes_both_artifacts(tiny_experiment, tmp_path):
    json_path = tmp_path / "metrics" / "evaluation.json"
    csv_path = tmp_path / "metrics" / "comparison.csv"
    save_results(tiny_experiment, json_path, csv_path)

    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload["model_results"]) == {"random", "popularity_count", "content_based"}
    # The dashboard is driven from the JSON alone, so the flat rows must be there too.
    assert payload["comparison"]
    assert {row["model"] for row in payload["comparison"]} == set(payload["model_results"])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for column in ("model", "k", "precision", "recall", "ndcg", "coverage"):
        assert column in header


def test_comparison_table_carries_tuned_parameters(tiny_experiment):
    table = comparison_table(tiny_experiment)
    assert all(isinstance(row["params"], str) for row in table)


def test_every_family_has_a_parameter_grid():
    for family in model_families():
        assert family in PARAM_GRIDS
        assert PARAM_GRIDS[family], f"{family} has an empty grid"


def test_tuning_selects_the_best_validation_candidate(data, splits, content):
    result = tune_family("popularity_bayes", splits.core, content, k=10)
    assert result.candidates
    assert result.best_params == result.best.params
    assert result.best.score == max(c.score for c in result.candidates)
    # The selected score must be reported on validation data, not test data.
    assert result.best.metrics["n_users"] == len(splits.core.test_sets())


def test_experiment_selected_params_match_the_tuning_winner(data, splits):
    result = run_experiment(data, splits, k_values=(10,), families=["popularity_bayes"], tune=True)
    tuning = result["tuning"]["popularity_bayes"]
    model_params = result["model_results"]["popularity_bayes"]["params"]
    assert model_params == tuning["best_params"]
    assert model_params in [c["params"] for c in tuning["candidates"]]
