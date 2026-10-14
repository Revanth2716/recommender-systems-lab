"""Metric definitions checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from recsys_lab.metrics import (
    catalog_coverage,
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_hand_computed():
    # 2 of the first 3 recommendations are relevant.
    assert precision_at_k([1, 4, 2], {1, 2, 3}, 3) == pytest.approx(2 / 3)
    # Truncating to K=2 leaves one hit out of two slots.
    assert precision_at_k([1, 4, 2], {1, 2, 3}, 2) == pytest.approx(1 / 2)


def test_recall_at_k_hand_computed():
    assert recall_at_k([1, 4, 2], {1, 2, 3}, 3) == pytest.approx(2 / 3)
    assert recall_at_k([1, 4, 2], {1, 2, 3}, 2) == pytest.approx(1 / 3)


def test_ndcg_at_k_hand_computed():
    # Hits at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg3 = sum(1.0 / math.log2(r + 1) for r in range(1, 4))
    assert ndcg_at_k([1, 4, 2], {1, 2, 3}, 3) == pytest.approx(dcg / idcg3)
    idcg2 = sum(1.0 / math.log2(r + 1) for r in range(1, 3))
    assert ndcg_at_k([1, 4], {1, 2, 3}, 2) == pytest.approx((1.0 / math.log2(2)) / idcg2)


def test_ndcg_rewards_earlier_hits():
    rel = {7}
    early = ndcg_at_k([7, 1, 2], rel, 3)
    late = ndcg_at_k([1, 2, 7], rel, 3)
    assert early > late > 0


def test_perfect_ranking_scores_one():
    rel = {1, 2, 3}
    assert precision_at_k([1, 2, 3], rel, 3) == 1.0
    assert recall_at_k([1, 2, 3], rel, 3) == 1.0
    assert ndcg_at_k([1, 2, 3], rel, 3) == pytest.approx(1.0)


def test_no_hits_scores_zero():
    assert precision_at_k([9, 8], {1, 2}, 2) == 0.0
    assert recall_at_k([9, 8], {1, 2}, 2) == 0.0
    assert ndcg_at_k([9, 8], {1, 2}, 2) == 0.0
    assert hit_rate_at_k([9, 8], {1, 2}, 2) == 0.0


def test_empty_relevant_set_is_zero_not_error():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0
    assert hit_rate_at_k([1, 2], set(), 2) == 0.0


def test_short_recommendation_list_is_not_padded():
    # Only two recommendations supplied for K=3: the empty slot must not count as
    # a hit, but precision divides by K, the conventional definition.
    assert precision_at_k([1, 2], {1, 2, 3, 4}, 3) == pytest.approx(2 / 3)
    assert recall_at_k([1, 2], {1, 2, 3, 4}, 3) == pytest.approx(2 / 4)


def test_k_must_be_positive():
    for fn in (precision_at_k, recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            fn([1], {1}, 0)


def test_evaluate_ranking_averages_over_users_with_held_out_items():
    recs = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
    test_sets = {0: {1, 2}, 1: {9}, 2: set()}
    out = evaluate_ranking(recs, test_sets, 3)
    assert out["n_users"] == 2  # user 2 has no held-out items and is skipped
    assert out["precision"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert out["recall"] == pytest.approx((1.0 + 0.0) / 2)


def test_catalog_coverage_counts_distinct_items():
    assert catalog_coverage({0: [1, 2], 1: [2, 3]}) == 3.0


def test_recall_and_hit_rate_are_monotone_in_k():
    """Recall and hit rate can only grow with K; precision has no such guarantee."""
    recs = [3, 1, 9, 4, 7]
    relevant = {1, 4, 9}
    recalls = [recall_at_k(recs, relevant, k) for k in range(1, 6)]
    hits = [hit_rate_at_k(recs, relevant, k) for k in range(1, 6)]
    assert recalls == sorted(recalls)
    assert hits == sorted(hits)
    # Precision at K=1 is 0 here but K=2 is 0.5, so monotonicity must not be asserted.
    assert precision_at_k(recs, relevant, 1) == 0.0
    assert precision_at_k(recs, relevant, 2) == pytest.approx(0.5)
