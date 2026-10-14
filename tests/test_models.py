"""Model contracts: shapes, masking, explanations and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from recsys_lab.metrics import evaluate_ranking
from recsys_lab.models import (
    ContentBasedRecommender,
    ItemItemCF,
    PopularityRecommender,
    RandomRecommender,
    SVDRecommender,
    make_model,
    model_families,
)
from recsys_lab.ranking import top_k_matrix

FAMILIES = [f for f in model_families() if f != "random"]


@pytest.fixture(scope="session")
def train_matrix(splits):
    return splits.final.train_matrix()


@pytest.fixture(scope="session")
def fitted(train_matrix, content):
    models = {}
    for family in model_families():
        model = make_model(family)
        model.fit(train_matrix, content)
        models[family] = model
    return models


@pytest.mark.parametrize("family", model_families())
def test_scores_are_a_finite_user_item_matrix(family, fitted, data):
    scores = fitted[family].score_all()
    assert scores.shape == (data.n_users, data.n_items)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("family", model_families())
def test_recommendations_exclude_seen_items(family, fitted, data):
    model = fitted[family]
    seen = model.seen_items()[0]
    recs = model.recommend(0, 10, seen)
    assert len(recs) == 10
    assert len(set(recs)) == 10
    assert not (set(recs) & seen)


@pytest.mark.parametrize("family", model_families())
def test_explanations_are_non_empty_strings(family, fitted, data):
    model = fitted[family]
    recs = model.recommend(0, 3, model.seen_items()[0])
    for item in recs:
        text = model.explain(0, item)
        assert isinstance(text, str)
        assert len(text) > 10


@pytest.mark.parametrize("family", FAMILIES)
def test_models_beat_the_random_floor(family, fitted, splits):
    """Every real model must out-rank the seeded random floor on the test tail."""
    test_sets = splits.final.test_sets()
    baseline_model = fitted["random"]
    baseline_recs = top_k_matrix(baseline_model.score_all(), 10, baseline_model.seen_items())
    baseline = evaluate_ranking(baseline_recs, test_sets, 10)["ndcg"]

    model = fitted[family]
    recs = top_k_matrix(model.score_all(), 10, model.seen_items())
    score = evaluate_ranking(recs, test_sets, 10)["ndcg"]
    assert score > baseline, f"{family} ({score:.4f}) did not beat random ({baseline:.4f})"


def test_popularity_count_ranks_the_most_rated_movie_first(train_matrix, content):
    model = PopularityRecommender(mode="count").fit(train_matrix, content)
    counts = np.asarray(train_matrix.getnnz(axis=0)).ravel()
    assert model.recommend(0, 1, set())[0] == int(np.argmax(counts))


def test_popularity_bayes_is_rank_invariant_across_users(fitted):
    scores = fitted["popularity_bayes"].score_all()
    np.testing.assert_allclose(scores[0], scores[1])


def test_random_baseline_is_reproducible(train_matrix, content):
    a = RandomRecommender(seed=7).fit(train_matrix, content).score_all()
    b = RandomRecommender(seed=7).fit(train_matrix, content).score_all()
    np.testing.assert_array_equal(a, b)


def test_svd_reconstruction_tracks_observed_ratings(train_matrix, content):
    """PureSVD should recover the dominant rating structure, not noise."""
    model = SVDRecommender(n_components=50).fit(train_matrix, content)
    recon = model.score_all()
    observed = train_matrix.nonzero()
    corr = np.corrcoef(recon[observed[0], observed[1]], train_matrix.data.astype(float))[0, 1]
    # A 50-factor reconstruction of a 93.7%-sparse matrix recovers roughly half the
    # variance of the observed entries; anything above 0.4 is dominant structure
    # rather than noise, which is all this test claims.
    assert corr > 0.4


def test_item_item_shrinkage_weakens_sparse_similarities(train_matrix, content):
    weak = ItemItemCF(shrink=1.0, neighbor_k=0).fit(train_matrix, content)
    strong = ItemItemCF(shrink=1000.0, neighbor_k=0).fit(train_matrix, content)
    assert np.abs(strong.similarity).sum() < np.abs(weak.similarity).sum()


def test_content_features_exist_for_every_item(content, data, train_matrix):
    assert len(content.titles) == data.n_items
    assert content.genre_matrix.shape[0] == data.n_items
    model = ContentBasedRecommender().fit(train_matrix, content)
    assert model.features.shape[0] == data.n_items
    assert len(model.feature_names) == model.features.shape[1]


@pytest.mark.parametrize("family", FAMILIES)
def test_tuned_params_round_trip_through_the_factory(family):
    model = make_model(family)
    assert isinstance(model.params, dict)
    clone = make_model(family, model.params)
    assert clone.params == model.params
    assert clone.label == model.label


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        make_model("not_a_model")
