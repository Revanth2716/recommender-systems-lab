"""The held-out protocol must not leak, must be temporal, and must be deterministic."""

from __future__ import annotations

import pandas as pd

from recsys_lab.split import temporal_leave_last_n, three_way_split


def _user_items(df: pd.DataFrame, user_idx: int) -> set[int]:
    rows = df[df["user_idx"] == user_idx]
    return {int(i) for i in rows["item_idx"]}


def test_train_and_test_are_disjoint(data):
    split = temporal_leave_last_n(data)
    pairs_train = set(zip(split.train["user_idx"], split.train["item_idx"], strict=True))
    pairs_test = set(zip(split.test["user_idx"], split.test["item_idx"], strict=True))
    assert not (pairs_train & pairs_test)
    assert len(pairs_train) + len(pairs_test) == len(data.ratings)


def test_every_user_has_train_and_test(data):
    split = temporal_leave_last_n(data)
    train_users = set(split.train["user_idx"])
    test_users = set(split.test["user_idx"])
    assert train_users == set(range(data.n_users))
    assert test_users == set(range(data.n_users))


def test_test_items_are_the_most_recent_per_user(data):
    split = temporal_leave_last_n(data, test_fraction=0.2, max_test=20)
    for user_idx, group in split.train.groupby("user_idx"):
        train_max = group["timestamp"].max()
        test_min = split.test[split.test["user_idx"] == user_idx]["timestamp"].min()
        assert train_max <= test_min


def test_split_is_deterministic(data):
    a = temporal_leave_last_n(data)
    b = temporal_leave_last_n(data)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_max_test_cap_is_respected(data):
    split = temporal_leave_last_n(data, max_test=5)
    assert split.test.groupby("user_idx").size().max() <= 5
    assert split.test.groupby("user_idx").size().min() >= 1


def test_training_matrix_matches_training_rows(data):
    split = temporal_leave_last_n(data)
    matrix = split.train_matrix()
    assert matrix.shape == (data.n_users, data.n_items)
    assert matrix.nnz == len(split.train)
    assert matrix.sum() == split.train["rating"].sum()


def test_three_way_split_nests_correctly(data):
    tw = three_way_split(data)

    # The test tail is the newest slice of the timeline.
    for user_idx in range(data.n_users):
        final_train = tw.final.train[tw.final.train["user_idx"] == user_idx]
        final_test = tw.final.test[tw.final.test["user_idx"] == user_idx]
        assert final_train["timestamp"].max() <= final_test["timestamp"].min()

    # The core fitting set is a strict subset of the final fitting set.
    core_pairs = set(zip(tw.core.train["user_idx"], tw.core.train["item_idx"], strict=True))
    final_pairs = set(zip(tw.final.train["user_idx"], tw.final.train["item_idx"], strict=True))
    assert core_pairs < final_pairs

    # Validation items are part of the final training set, never of the test set.
    val_pairs = set(zip(tw.core.test["user_idx"], tw.core.test["item_idx"], strict=True))
    test_pairs = set(zip(tw.final.test["user_idx"], tw.final.test["item_idx"], strict=True))
    assert val_pairs <= final_pairs
    assert not (val_pairs & test_pairs)


def test_three_way_validation_is_newer_than_core_training(data):
    tw = three_way_split(data)
    for user_idx, group in tw.core.train.groupby("user_idx"):
        val = tw.core.test[tw.core.test["user_idx"] == user_idx]
        assert group["timestamp"].max() <= val["timestamp"].min()


def test_split_stats_add_up(data):
    tw = three_way_split(data)
    stats = tw.stats()
    assert stats["core"]["n_train"] + stats["validation_items"] == stats["final"]["n_train"]
    assert stats["final"]["n_train"] + stats["test_items"] == len(data.ratings)
