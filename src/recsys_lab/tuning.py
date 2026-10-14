"""Hyperparameter selection.

Candidates are scored on the *validation tail* of the training period (see
``split.three_way_split``). The test tail is never used here, so the numbers in
the final comparison table are honest held-out numbers rather than best-of-grid
values cherry-picked on the test set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from recsys_lab.metrics import evaluate_ranking
from recsys_lab.models import PARAM_GRIDS, Content, make_model
from recsys_lab.ranking import top_k_matrix
from recsys_lab.split import Split

TUNING_METRIC = "ndcg"
TUNING_K = 10


@dataclass
class Candidate:
    params: dict[str, object]
    metrics: dict[str, float]

    @property
    def score(self) -> float:
        return float(self.metrics[TUNING_METRIC])


@dataclass
class TuningResult:
    name: str
    best_params: dict[str, object]
    candidates: list[Candidate] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def best(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.score)


def tune_family(name: str, split: Split, content: Content, k: int = TUNING_K) -> TuningResult:
    """Fit every candidate config for one model family on ``split`` and rank them."""
    train_matrix = split.train_matrix()
    test_sets = split.test_sets()
    result = TuningResult(name=name, best_params={})
    t0 = time.perf_counter()

    for params in PARAM_GRIDS[name]:
        model = make_model(name, params)
        model.fit(train_matrix, content)
        recs = top_k_matrix(model.score_all(), k, model.seen_items())
        metrics = evaluate_ranking(recs, test_sets, k)
        result.candidates.append(
            Candidate(params=dict(params), metrics={kk: round(v, 6) for kk, v in metrics.items()})
        )

    result.seconds = round(time.perf_counter() - t0, 3)
    result.best_params = dict(result.best.params)
    return result


def tune_all(
    families: list[str], split: Split, content: Content, k: int = TUNING_K
) -> dict[str, TuningResult]:
    """Tune every requested model family."""
    return {name: tune_family(name, split, content, k) for name in families}
