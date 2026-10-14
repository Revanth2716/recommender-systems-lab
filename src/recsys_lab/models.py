"""Recommender models.

Every model implements a small common interface::

    model.fit(train_matrix, content)   # scipy CSR user x item training ratings
    scores = model.score_all()         # dense (n_users, n_items) score matrix

Having ``score_all`` means all models are evaluated under exactly the same
all-unrated-items ranking protocol: each user's score vector is masked to hide
items they already rated in training, and the top-K entries are taken.

Models
------
PopularityRecommender
    Non-personalized baseline. ``bayes`` mode uses the IMDb-style weighted rating
    ``(v/(v+m))R + (m/(v+m))C``; ``count`` mode ranks by raw rating count.
ContentBasedRecommender
    TF-IDF over movie titles concatenated with genre indicators; a user profile is
    a relevance-weighted centroid of the items they liked, scored by cosine
    similarity.
ItemItemCF
    Item-item collaborative filtering (Sarwar et al., 2001) — cosine similarity on
    user-mean-centered ratings with shrinkage and a neighbour cap, scored by a
    similarity-weighted sum of the user's centered ratings.
SVDRecommender
    PureSVD (Cremonesi et al., 2010): truncated SVD of the rating matrix, scoring
    by the reconstruction ``U S Vᵀ``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .ranking import top_k_items


@dataclass
class Content:
    """Item-side content features shared by the models.

    Attributes
    ----------
    titles: display titles, used in explanations.
    cleaned_titles: titles with the trailing ``(year)`` removed, used for TF-IDF.
    genre_names: the 19 MovieLens genre column names.
    genre_matrix: ``(n_items, n_genres)`` binary indicator matrix.
    """

    titles: list[str]
    cleaned_titles: list[str]
    genre_names: list[str]
    genre_matrix: np.ndarray

    @property
    def genres(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.genre_matrix.astype(np.float32))


def build_content(items: pd.DataFrame, genre_names: list[str]) -> Content:
    """Build :class:`Content` from the dataset's item metadata table."""
    titles = [str(t) for t in items["title"]]
    cleaned = [t.rsplit(" (", 1)[0] if t.endswith(")") else t for t in titles]
    genre_matrix = items[list(genre_names)].to_numpy(dtype=np.float32)
    return Content(
        titles=titles,
        cleaned_titles=cleaned,
        genre_names=list(genre_names),
        genre_matrix=genre_matrix,
    )


def _user_means(matrix: sparse.csr_matrix) -> np.ndarray:
    """Per-user mean of observed ratings; falls back to the global mean."""
    m = matrix.tocsr()
    counts = np.asarray(m.getnnz(axis=1)).ravel()
    sums = np.asarray(m.sum(axis=1)).ravel()
    global_mean = float(m.sum() / m.nnz) if m.nnz else 0.0
    means = np.full(m.shape[0], global_mean, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means


class Recommender:
    """Common interface for all recommenders in the lab."""

    kind = "generic"

    def __init__(self) -> None:
        self.name = "recommender"
        self.label = "Recommender"
        self.params: dict[str, object] = {}
        self._scores: np.ndarray | None = None
        self._train: sparse.csr_matrix | None = None
        self._content: Content | None = None

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> Recommender:
        raise NotImplementedError

    def score_all(self) -> np.ndarray:
        """Dense ``(n_users, n_items)`` score matrix; higher is better."""
        if self._scores is None:
            raise RuntimeError(f"{self.name} must be fitted before scoring")
        return self._scores

    def recommend(self, user_idx: int, k: int = 10, exclude: set[int] | None = None) -> list[int]:
        return top_k_items(self.score_all()[user_idx], k, exclude)

    def seen_items(self) -> dict[int, set[int]]:
        """Training interactions per user, used to mask already-rated items."""
        if self._train is None:
            return {}
        train = self._train.tocsr()
        return {
            u: {int(i) for i in train.indices[train.indptr[u] : train.indptr[u + 1]]}
            for u in range(train.shape[0])
        }

    def explain(self, user_idx: int, item_idx: int) -> str:
        """Short human-readable reason for a recommendation."""
        return "Recommended by a non-personalized signal."


# --------------------------------------------------------------------------- #
# Popularity
# --------------------------------------------------------------------------- #


class PopularityRecommender(Recommender):
    """Non-personalized popularity baseline."""

    kind = "popularity"

    def __init__(self, mode: str = "bayes", prior_count: float = 25.0) -> None:
        super().__init__()
        if mode not in {"bayes", "count"}:
            raise ValueError("mode must be 'bayes' or 'count'")
        self.mode = mode
        self.prior_count = prior_count
        self.name = f"popularity_{mode}"
        self.label = (
            "Popularity (rating count)"
            if mode == "count"
            else f"Popularity (Bayesian, m={prior_count:g})"
        )
        self.params = {"mode": mode, "prior_count": prior_count}
        self.counts: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.item_scores: np.ndarray | None = None
        self.global_mean: float = 0.0

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> PopularityRecommender:
        m = train_matrix.tocsr()
        self._train = m
        self._content = content
        counts = np.asarray(m.getnnz(axis=0)).ravel().astype(np.float64)
        sums = np.asarray(m.sum(axis=0)).ravel()
        self.global_mean = float(m.sum() / m.nnz) if m.nnz else 0.0
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
        self.counts, self.means = counts, means

        if self.mode == "count":
            score = counts
        else:
            v, R, C, prior = counts, means, self.global_mean, self.prior_count
            weight = v / (v + prior)
            score = weight * R + (1.0 - weight) * C

        # Tiny deterministic tiebreak on count keeps ordering stable for ties.
        self.item_scores = score + 1e-9 * counts
        self._scores = np.tile(self.item_scores, (m.shape[0], 1))
        return self

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self.counts is not None and self.means is not None
        n = int(self.counts[item_idx])
        mean = float(self.means[item_idx])
        if self.mode == "count":
            return f"Rated by {n:,} users - among the most-rated titles in the catalog."
        return (
            f"Bayesian weighted rating {self.item_scores[item_idx]:.2f} from {n:,} ratings "
            f"(avg {mean:.2f}); prior m={self.prior_count:g}."
        )


# --------------------------------------------------------------------------- #
# Content-based
# --------------------------------------------------------------------------- #


class ContentBasedRecommender(Recommender):
    """Content-based filtering over genres + TF-IDF of titles.

    ``profile_scheme`` controls how a user's ratings are turned into a profile
    weight for each item they rated: ``centered`` (rating minus the user's mean,
    clipped at 0), ``liked`` (1 for rating >= 4) or ``raw`` (the rating itself).
    """

    kind = "content"
    SCHEMES = ("centered", "liked", "raw")

    def __init__(
        self,
        use_title: bool = True,
        use_genre: bool = True,
        profile_scheme: str = "centered",
    ) -> None:
        super().__init__()
        if profile_scheme not in self.SCHEMES:
            raise ValueError(f"profile_scheme must be one of {self.SCHEMES}")
        if not (use_title or use_genre):
            raise ValueError("at least one of use_title/use_genre must be True")
        self.name = "content_based"
        blocks = [b for b, on in (("title", use_title), ("genre", use_genre)) if on]
        self.label = f"Content-based ({'+'.join(blocks)}, {profile_scheme})"
        self.params = {
            "use_title": use_title,
            "use_genre": use_genre,
            "profile_scheme": profile_scheme,
        }
        self.use_title = use_title
        self.use_genre = use_genre
        self.profile_scheme = profile_scheme
        self.features: sparse.csr_matrix | None = None
        self.feature_names: list[str] = []
        self._profiles: sparse.csr_matrix | None = None

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> ContentBasedRecommender:
        m = train_matrix.tocsr()
        self._train = m
        self._content = content

        blocks: list[sparse.csr_matrix] = []
        names: list[str] = []
        if self.use_title:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                sublinear_tf=True,
                token_pattern=r"[A-Za-z][A-Za-z'\-]+",
            )
            blocks.append(normalize(vectorizer.fit_transform(content.cleaned_titles)))
            names += list(vectorizer.get_feature_names_out())
        if self.use_genre:
            blocks.append(normalize(content.genres))
            names += list(content.genre_names)

        self.feature_names = names
        self.features = blocks[0] if len(blocks) == 1 else sparse.hstack(blocks, format="csr")

        self._profiles = sparse.csr_matrix(self._build_profiles(m))
        item_user = self.features @ self._profiles.T  # (n_items, n_users)
        if sparse.issparse(item_user):
            item_user = item_user.toarray()
        self._scores = np.asarray(item_user, dtype=np.float64).T
        return self

    def _weights(self, ratings: np.ndarray, user_mean: float) -> np.ndarray:
        if self.profile_scheme == "raw":
            return ratings
        if self.profile_scheme == "liked":
            return (ratings >= 4).astype(np.float64)
        weights = np.clip(ratings - user_mean, 0.0, None)
        if not np.any(weights > 0):
            weights = np.clip(ratings - ratings.min() + 1e-6, 0.0, None)
        return weights

    def _build_profiles(self, m: sparse.csr_matrix) -> np.ndarray:
        assert self.features is not None
        features = self.features
        user_means = _user_means(m)
        rows = m.tocsr()
        profiles = np.zeros((m.shape[0], features.shape[1]), dtype=np.float64)
        for u in range(m.shape[0]):
            start, end = rows.indptr[u], rows.indptr[u + 1]
            items = rows.indices[start:end]
            if items.size == 0:
                continue
            weights = self._weights(rows.data[start:end].astype(np.float64), user_means[u])
            if weights.sum() <= 0:
                continue
            profiles[u] = np.asarray(features[items].T @ (weights / weights.sum())).ravel()
        return normalize(sparse.csr_matrix(profiles)).toarray()

    def top_features(self, user_idx: int, item_idx: int, n: int = 4) -> list[str]:
        """Features the item shares with the user's profile, ranked by contribution."""
        assert self.features is not None and self._profiles is not None
        assert self._content is not None
        item_vec = self.features.getrow(item_idx)
        profile_vec = self._profiles.getrow(user_idx)
        contribution = np.asarray(item_vec.multiply(profile_vec).todense()).ravel()
        order = np.argsort(-contribution)
        picked = [self.feature_names[i] for i in order if contribution[i] > 0][: n + 6]
        genres = [f for f in picked if f in self._content.genre_names and f != "unknown"]
        tokens = [f for f in picked if f not in self._content.genre_names]
        return (genres + tokens)[:n]

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self._content is not None
        feats = self.top_features(user_idx, item_idx)
        title = self._content.titles[item_idx]
        if feats:
            return (
                f"'{title}' shares {', '.join(feats)} with titles this user rated "
                f"above their personal average."
            )
        return f"'{title}' matches the user's title-token profile."


# --------------------------------------------------------------------------- #
# Collaborative filtering: item-item cosine
# --------------------------------------------------------------------------- #


class ItemItemCF(Recommender):
    """Item-item collaborative filtering on user-mean-centered ratings."""

    kind = "collaborative"

    def __init__(self, shrink: float = 100.0, neighbor_k: int = 200) -> None:
        super().__init__()
        self.name = "item_item_cf"
        self.label = f"Item-item CF (shrink={shrink:g}, k={neighbor_k})"
        self.params = {"shrink": shrink, "neighbor_k": neighbor_k}
        self.shrink = shrink
        self.neighbor_k = neighbor_k
        self.similarity: np.ndarray | None = None
        self._centered: sparse.csr_matrix | None = None

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> ItemItemCF:
        m = train_matrix.tocsr()
        self._train = m
        self._content = content

        user_means = _user_means(m)
        centered = m.astype(np.float64).copy()
        rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        centered.data -= user_means[rows]
        self._centered = centered.tocsr()

        item_vectors = normalize(self._centered.T.tocsr())
        sim = (item_vectors @ item_vectors.T).toarray()
        np.fill_diagonal(sim, 0.0)

        # Shrinkage dampens similarities estimated from few co-rating users.
        binary = (m > 0).astype(np.float64)
        co = np.asarray((binary.T @ binary).todense())
        sim = sim * (co / (co + self.shrink))

        if self.neighbor_k and 0 < self.neighbor_k < sim.shape[1]:
            keep = np.argpartition(-sim, self.neighbor_k - 1, axis=1)[:, : self.neighbor_k]
            mask = np.zeros_like(sim, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            sim = np.where(mask, sim, 0.0)

        self.similarity = sim
        # Textbook item-based prediction (Sarwar et al., 2001): similarity-weighted
        # sum of the user's centered ratings. Unlike a similarity-normalised score,
        # this keeps the magnitude of the evidence, so an item supported by one
        # weak neighbour cannot outrank an item supported by many strong ones.
        self._scores = np.asarray(self._centered @ sim.T, dtype=np.float64)
        return self

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self.similarity is not None and self._centered is not None
        assert self._content is not None and self._train is not None
        centered = self._centered.getrow(user_idx)
        rated, centered_ratings = centered.indices, centered.data.astype(np.float64)
        if rated.size == 0:
            return "No training interactions for this user."
        raw_row = self._train.getrow(user_idx)
        raw_ratings = dict(zip(raw_row.indices, raw_row.data, strict=True))
        sims = self.similarity[item_idx, rated]
        order = np.argsort(-np.abs(sims))
        # Only items the user rated above their own average actually push the score up.
        liked = [
            (int(rated[o]), float(sims[o]), float(raw_ratings.get(int(rated[o]), 0.0)))
            for o in order
            if centered_ratings[o] > 0
        ][:3]
        if not liked:
            return "Recommended from item-item co-rating patterns across all users."
        parts = [
            f"'{self._content.titles[i]}' ({r:.0f} stars, similarity {s:+.2f})" for i, s, r in liked
        ]
        return "Because you rated " + ", ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Collaborative filtering: PureSVD
# --------------------------------------------------------------------------- #


class SVDRecommender(Recommender):
    """PureSVD latent-factor model (Cremonesi et al., 2010)."""

    kind = "collaborative"

    def __init__(self, n_components: int = 50, centered: bool = False, seed: int = 0) -> None:
        super().__init__()
        self.n_components = n_components
        self.centered = centered
        self.seed = seed
        self.name = "svd_cf"
        self.label = f"PureSVD (k={n_components}, {'mean-centered' if centered else 'raw'})"
        self.params = {"n_components": n_components, "centered": centered}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> SVDRecommender:
        m = train_matrix.tocsr().astype(np.float64)
        self._train = m
        self._content = content
        means = _user_means(m)

        if self.centered:
            rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
            m = m.copy()
            m.data -= means[rows]

        k = int(min(self.n_components, min(m.shape) - 1))
        rng = np.random.default_rng(self.seed)
        u, s, vt = svds(m, k=k, random_state=rng)
        order = np.argsort(-s)
        self.singular_values = s[order]
        self.user_factors = u[:, order] * s[order]
        self.item_factors = vt[order].T
        scores = self.user_factors @ self.item_factors.T
        # Adding the user mean back is rank-invariant within a user; it only makes
        # the displayed score interpretable on the original 1-5 star scale.
        self._scores = scores + means[:, None] if self.centered else scores
        return self

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self.item_factors is not None and self.user_factors is not None
        assert self._content is not None and self._train is not None
        item_vec = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1.0
        sims = (self.item_factors @ item_vec) / (norms * np.linalg.norm(item_vec) + 1e-12)
        rated = self._train.getrow(user_idx).indices
        mask = np.full(sims.shape, -np.inf)
        mask[rated] = sims[rated]
        top = [int(i) for i in np.argsort(-mask)[:2] if np.isfinite(mask[i])]
        if top:
            names = ", ".join(f"'{self._content.titles[i]}'" for i in top)
            return (
                f"The {self.n_components}-dimensional latent factors place this near "
                f"{names}, which this user rated highly."
            )
        return (
            f"Latent-factor score {self._scores[user_idx, item_idx]:.2f} from a "
            f"{self.n_components}-dimensional SVD of the rating matrix."
        )


# --------------------------------------------------------------------------- #
# Baseline: random
# --------------------------------------------------------------------------- #


class RandomRecommender(Recommender):
    """Seeded uniform-random scores: the floor the other models must beat."""

    kind = "baseline"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.name = "random"
        self.label = "Random (seeded floor)"
        self.params = {"seed": seed}
        self.seed = seed

    def fit(self, train_matrix: sparse.csr_matrix, content: Content) -> RandomRecommender:
        self._train = train_matrix.tocsr()
        self._content = content
        rng = np.random.default_rng(self.seed)
        self._scores = rng.random((self._train.shape[0], self._train.shape[1]))
        return self

    def explain(self, user_idx: int, item_idx: int) -> str:
        return "Uniformly random pick - only useful as a floor for the ranking metrics."


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

PARAM_GRIDS: dict[str, list[dict[str, object]]] = {
    "popularity_bayes": [{"mode": "bayes", "prior_count": m} for m in (5.0, 25.0, 100.0)],
    "popularity_count": [{"mode": "count"}],
    "content_based": [
        {"use_title": t, "use_genre": g, "profile_scheme": s}
        for t, g in ((True, False), (False, True), (True, True))
        for s in ("centered", "liked", "raw")
    ],
    "item_item_cf": [{"shrink": s, "neighbor_k": k} for s in (20.0, 100.0) for k in (50, 200)],
    "svd_cf": [{"n_components": k, "centered": c} for k in (20, 50, 100) for c in (False, True)],
    "random": [{"seed": 0}],
}


def make_model(name: str, params: dict[str, object] | None = None) -> Recommender:
    """Instantiate a model by registry name with optional hyperparameters."""
    params = dict(params or {})
    if name == "popularity_bayes":
        params.setdefault("mode", "bayes")
        params.setdefault("prior_count", 25.0)
        return PopularityRecommender(**params)  # type: ignore[arg-type]
    if name == "popularity_count":
        return PopularityRecommender(mode="count")
    if name == "content_based":
        return ContentBasedRecommender(**params)  # type: ignore[arg-type]
    if name == "item_item_cf":
        return ItemItemCF(**params)  # type: ignore[arg-type]
    if name == "svd_cf":
        return SVDRecommender(**params)  # type: ignore[arg-type]
    if name == "random":
        params.setdefault("seed", 0)
        return RandomRecommender(**params)  # type: ignore[arg-type]
    raise KeyError(f"unknown model '{name}'")


def model_families() -> list[str]:
    """Model families in the default comparison, in reporting order."""
    return [
        "random",
        "popularity_count",
        "popularity_bayes",
        "content_based",
        "item_item_cf",
        "svd_cf",
    ]
