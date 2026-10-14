"""Held-out evaluation protocol.

Protocol: **per-user temporal leave-last-n**.

For every user we sort their rated items by timestamp (ties broken
deterministically by item id) and hold out the most recent ``n_u`` interactions,
where ``n_u = clip(ceil(fraction * n_u), 1, max_test)``. Everything older forms
the training set.

Properties that make this defensible:
  * It is a *forward-in-time* split, so a model is never trained on future
    interactions relative to the items it is asked to rank.
  * Every user contributes to both sides (``min_train`` is enforced), so per-user
    metrics are defined for all 943 users.
  * It is deterministic: no random draws, so re-running the pipeline reproduces
    the exact same split.
  * ``three_way_split`` carves a validation tail out of the training period, so
    hyperparameters are chosen on validation data and the test split is touched
    exactly once, for the final reported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .data import MovieLensData


@dataclass
class Split:
    """A train/test split of the rating matrix in positional indices."""

    train: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def train_matrix(self) -> sparse.csr_matrix:
        """User x item matrix of training ratings (unobserved entries are 0)."""
        return sparse.csr_matrix(
            (
                self.train["rating"].to_numpy(dtype=np.float32),
                (self.train["user_idx"].to_numpy(), self.train["item_idx"].to_numpy()),
            ),
            shape=(self.n_users, self.n_items),
        )

    def test_sets(self) -> dict[int, set[int]]:
        """Map user index -> set of held-out item indices."""
        out: dict[int, set[int]] = {}
        for u, group in self.test.groupby("user_idx"):
            out[int(u)] = {int(i) for i in group["item_idx"]}
        return out

    def seen_train(self) -> dict[int, set[int]]:
        """Map user index -> items observed in training (the ranking mask)."""
        out: dict[int, set[int]] = {}
        for u, group in self.train.groupby("user_idx"):
            out[int(u)] = {int(i) for i in group["item_idx"]}
        return out

    def stats(self) -> dict[str, float]:
        return {
            "n_train": len(self.train),
            "n_test": len(self.test),
            "test_fraction": round(len(self.test) / max(len(self.train) + len(self.test), 1), 4),
            "mean_test_per_user": round(len(self.test) / self.n_users, 3),
            "min_test_per_user": int(self.test.groupby("user_idx").size().min()),
        }


def _per_user_leave_last(
    ratings: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    fraction: float,
    max_test: int,
    min_train: int,
) -> Split:
    """Deterministic per-user temporal split of an already-indexed rating table."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    ordered = ratings.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)

    counts = ordered.groupby("user_id")["rating"].transform("size").to_numpy()
    rank_desc = ordered.groupby("user_id").cumcount(ascending=False).to_numpy()

    n_test = np.clip(np.ceil(fraction * counts).astype(int), 1, max_test)
    n_test = np.minimum(n_test, np.maximum(counts - min_train, 1))
    is_test = rank_desc < n_test

    train = ordered.loc[~is_test].copy()
    test = ordered.loc[is_test].copy()
    train["is_test"] = False
    test["is_test"] = True

    missing = n_users - train["user_idx"].nunique()
    if missing:
        raise AssertionError(f"{missing} users have no training interactions")

    return Split(train=train, test=test, n_users=n_users, n_items=n_items)


def temporal_leave_last_n(
    data: MovieLensData,
    *,
    test_fraction: float = 0.2,
    max_test: int = 20,
    min_train: int = 1,
) -> Split:
    """Build a deterministic per-user temporal leave-last-n split."""
    return _per_user_leave_last(
        data.ratings,
        n_users=data.n_users,
        n_items=data.n_items,
        fraction=test_fraction,
        max_test=max_test,
        min_train=min_train,
    )


@dataclass
class ThreeWay:
    """Nested temporal splits.

    ``core``     fit on ``core.train``, tune on ``core.test`` (the validation tail).
    ``final``    fit on ``final.train`` (core + validation), report on ``final.test``
                 (the test tail, touched exactly once).
    """

    core: Split
    final: Split

    def stats(self) -> dict[str, object]:
        return {
            "core": self.core.stats(),
            "validation_items": len(self.core.test),
            "final": self.final.stats(),
            "test_items": len(self.final.test),
        }


def three_way_split(
    data: MovieLensData,
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.25,
    max_test: int = 20,
) -> ThreeWay:
    """Carve a validation tail out of the training period, then a test tail.

    The most recent 20% of each user's timeline becomes the test set. Of the
    remaining training period, the most recent 25% (about 20% of the whole
    timeline) becomes the validation set used to choose hyperparameters. Reported
    models are refit on train+validation and evaluated once on the test set.
    """
    outer = temporal_leave_last_n(data, test_fraction=test_fraction, max_test=max_test, min_train=1)
    inner = _per_user_leave_last(
        outer.train,
        n_users=data.n_users,
        n_items=data.n_items,
        fraction=val_fraction,
        max_test=max_test,
        min_train=1,
    )
    # `inner.train` is the core fitting set, `inner.test` is the validation tail,
    # and `outer.train` (core + validation) is the final fitting set.
    return ThreeWay(core=inner, final=outer)
