"""Top-K selection: ordering, exclusion and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from recsys_lab.ranking import top_k_items, top_k_matrix


def test_returns_highest_scoring_items_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k_items(scores, 2) == [1, 3]
    assert top_k_items(scores, 4) == [1, 3, 2, 0]


def test_excluded_items_never_appear():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert 1 not in top_k_items(scores, 3, exclude={1})


def test_k_larger_than_catalog_is_clamped():
    scores = np.array([0.1, 0.9])
    assert len(top_k_items(scores, 10)) == 2


def test_non_positive_k_returns_empty():
    assert top_k_items(np.array([1.0, 2.0]), 0) == []


def test_ties_break_on_item_index_for_determinism():
    scores = np.zeros(5)
    assert top_k_items(scores, 3) == [0, 1, 2]


def test_out_of_range_exclusions_are_ignored():
    scores = np.array([0.2, 0.8])
    assert top_k_items(scores, 2, exclude={99, -3}) == [1, 0]


def test_matrix_version_applies_per_user_exclusions():
    matrix = np.array([[0.5, 0.9, 0.1], [0.7, 0.2, 0.6]])
    out = top_k_matrix(matrix, 2, {0: {1}})
    assert out[0][0] != 1
    assert out[1] == [0, 2]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_top_k_is_reproducible(seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(50)
    assert top_k_items(scores, 5) == top_k_items(scores, 5)
