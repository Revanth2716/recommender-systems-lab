"""Helpers for turning score vectors into ranked top-K recommendation lists."""

from __future__ import annotations

import numpy as np


def top_k_items(scores: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
    """Return the indices of the ``k`` highest-scoring items.

    Items in ``exclude`` (typically the user's training interactions) are masked
    to negative infinity so they can never be recommended. Ties are broken by
    ascending item index for determinism.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if exclude:
        scores = scores.copy()
        idx = np.fromiter((i for i in exclude if 0 <= i < scores.size), dtype=np.int64)
        if idx.size:
            scores[idx] = -np.inf

    n = scores.size
    k = int(min(k, n))
    if k <= 0:
        return []

    # Partial selection is O(n); full sort is unnecessary for n_items = 1682 but
    # keeps behaviour identical for tiny inputs.
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = np.lexsort((candidates, -scores[candidates]))
    return [int(i) for i in candidates[order]]


def top_k_matrix(
    score_matrix: np.ndarray, k: int, seen: dict[int, set[int]] | None = None
) -> dict[int, list[int]]:
    """Top-K item lists for every user in a ``(n_users, n_items)`` score matrix."""
    seen = seen or {}
    return {
        u: top_k_items(score_matrix[u], k, seen.get(u, set())) for u in range(score_matrix.shape[0])
    }
