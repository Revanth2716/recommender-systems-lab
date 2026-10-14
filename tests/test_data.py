"""The dataset loader must reproduce the exact published MovieLens 100K contract."""

from __future__ import annotations

import pytest

from recsys_lab.data import (
    EXPECTED_N_GENRES,
    EXPECTED_N_ITEMS,
    EXPECTED_N_RATINGS,
    EXPECTED_N_USERS,
    GENRES,
    load_items,
    load_ratings,
    load_users,
    sha256_of,
    validate_ml100k,
)


def test_validation_matches_published_counts(data):
    stats = validate_ml100k(data)
    assert stats["n_ratings"] == EXPECTED_N_RATINGS
    assert stats["n_users"] == EXPECTED_N_USERS
    assert stats["n_items"] == EXPECTED_N_ITEMS
    assert stats["min_rating"] == 1
    assert stats["max_rating"] == 5
    # MovieLens 100K is famously 93.7% sparse.
    assert stats["sparsity"] == pytest.approx(0.937, abs=0.001)


def test_rating_scale_is_one_to_five(data):
    assert data.ratings["rating"].between(1, 5).all()
    assert data.ratings["rating"].dtype.kind == "i"


def test_genre_matrix_shape_and_values(data):
    assert data.genre_matrix.shape == (data.n_items, EXPECTED_N_GENRES)
    assert len(GENRES) == EXPECTED_N_GENRES
    assert set(data.genre_matrix.ravel().tolist()) <= {0.0, 1.0}
    # Every movie has at least one genre flag (possibly "unknown").
    assert (data.genre_matrix.sum(axis=1) >= 1).all()


def test_indices_are_consistent(data):
    for raw_id, pos in data.item_index.items():
        assert data.item_ids[pos] == raw_id
    for raw_id, pos in data.user_index.items():
        assert data.user_ids[pos] == raw_id
    assert data.ratings["user_idx"].max() == data.n_users - 1
    assert data.ratings["item_idx"].max() == data.n_items - 1


def test_titles_and_metadata_are_populated(data):
    assert data.items["title"].notna().all()
    assert (data.items["title"].str.len() > 0).all()
    assert data.items["genre_list"].apply(lambda g: isinstance(g, list)).all()
    # A well-known title must be present, proving we parsed the real u.item file.
    assert data.items["title"].str.contains("Toy Story", regex=False).any()
    assert data.users["occupation"].notna().all()


def test_loaders_agree_with_the_indexed_dataset(data, raw_dir):
    assert len(load_ratings(raw_dir)) == EXPECTED_N_RATINGS
    assert len(load_items(raw_dir)) == EXPECTED_N_ITEMS
    assert len(load_users(raw_dir)) == EXPECTED_N_USERS
    assert load_items(raw_dir)["item_id"].is_unique


def test_validation_rejects_a_corrupted_dataset(data):
    import dataclasses

    broken = dataclasses.replace(data, ratings=data.ratings.iloc[:-1])
    with pytest.raises(AssertionError):
        validate_ml100k(broken)


def test_sha256_is_stable_for_the_archive(raw_dir):
    archive = raw_dir.parent / "ml-100k.zip"
    if not archive.exists():
        pytest.skip("archive not retained")
    digest = sha256_of(archive)
    assert len(digest) == 64
    assert digest == sha256_of(archive)
