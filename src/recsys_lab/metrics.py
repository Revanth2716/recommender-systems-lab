"""Ranking metrics for top-K recommendation: Precision@K, Recall@K, NDCG@K.

All metrics treat relevance as binary (an item is relevant iff the user actually
interacted with it in the held-out set) and are computed per user, then averaged
over users that have at least one held-out item. Users with no held-out items are
skipped rather than counted as zeros, which would silently deflate the scores.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def _as_set(items: Iterable[int]) -> set[int]:
    return {int(i) for i in items}


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(recommended)[:k]
    if not top:
        return 0.0
    rel = _as_set(relevant)
    return sum(1 for i in top if i in rel) / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of held-out relevant items recovered in the top-K list."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for i in top if i in rel) / len(rel)


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K with binary gains.

    DCG@K = sum_i rel_i / log2(i + 1), i being the 1-based rank.
    IDCG@K is the DCG of the best possible ordering (all relevant items first).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _as_set(relevant)
    if not rel:
        return 0.0

    top = list(recommended)[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(top, start=1) if item in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    recommendations: dict[int, Sequence[int]],
    test_sets: dict[int, Iterable[int]],
    k: int,
) -> dict[str, float]:
    """Average Precision@K, Recall@K and NDCG@K over users with held-out items."""
    p, r, n = [], [], []
    for user, relevant in test_sets.items():
        rel = _as_set(relevant)
        if not rel:
            continue
        recs = recommendations.get(user, [])
        p.append(precision_at_k(recs, rel, k))
        r.append(recall_at_k(recs, rel, k))
        n.append(ndcg_at_k(recs, rel, k))

    if not p:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "n_users": 0}
    return {
        "precision": float(np.mean(p)),
        "recall": float(np.mean(r)),
        "ndcg": float(np.mean(n)),
        "n_users": len(p),
    }


def hit_rate_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary hit rate (1.0 if any held-out item is in the top-K list)."""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(i in rel for i in list(recommended)[:k]) else 0.0


def catalog_coverage(recommendations: dict[int, Sequence[int]]) -> float:
    """Share of the catalog that appears at least once across all users' lists."""
    seen = {int(i) for recs in recommendations.values() for i in recs}
    return float(len(seen))
