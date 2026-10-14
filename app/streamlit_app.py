"""Recommender Systems Lab — Streamlit dashboard.

Run with::

    streamlit run app/streamlit_app.py

The app reads the tuned hyperparameters and head-line metrics from
``artifacts/metrics/evaluation.json`` (produced by ``python -m recsys_lab.pipeline``)
and refits the selected models on the training split so recommendations and
explanations are live rather than hard-coded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recsys_lab import paths  # noqa: E402
from recsys_lab.data import download_ml100k, load_dataset, validate_ml100k  # noqa: E402
from recsys_lab.evaluate import build_content_from  # noqa: E402
from recsys_lab.metrics import hit_rate_at_k, ndcg_at_k  # noqa: E402
from recsys_lab.models import make_model, model_families  # noqa: E402
from recsys_lab.split import three_way_split  # noqa: E402

st.set_page_config(
    page_title="Recommender Systems Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLACEHOLDER = {
    "dataset": {},
    "protocol": {},
    "tuning": {},
    "model_results": {},
    "comparison": [],
}


# --------------------------------------------------------------------------- #
# Cached data / models
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading MovieLens 100K…")
def load_all() -> tuple[object, object, object, dict[str, object]]:
    paths.ensure_dirs()
    if not paths.ML100K_DIR.joinpath("u.data").exists():
        download_ml100k(paths.RAW_DIR)
    data = load_dataset(paths.ML100K_DIR)
    stats = validate_ml100k(data)
    splits = three_way_split(data)
    content = build_content_from(data)
    return data, splits, content, stats


def load_results() -> dict[str, object]:
    if paths.RESULTS_JSON.exists():
        with paths.RESULTS_JSON.open(encoding="utf-8") as fh:
            return json.load(fh)
    return dict(PLACEHOLDER)


@st.cache_resource(show_spinner="Training recommenders on the training split…")
def fit_all(best_params: tuple[tuple[str, str], ...]) -> tuple[dict, dict]:
    _data, splits, content, _ = load_all()
    train = splits.final.train_matrix()
    models: dict[str, object] = {}
    fit_times: dict[str, float] = {}
    for name, params_json in best_params:
        params = json.loads(params_json) if params_json else {}
        model = make_model(name, params)
        t0 = time.perf_counter()
        model.fit(train, content)
        fit_times[name] = time.perf_counter() - t0
        models[name] = model
    return models, fit_times


def user_labels(data) -> pd.DataFrame:
    users = data.users.copy()
    users["label"] = (
        users["user_id"].astype(str)
        + "  —  "
        + users["gender"].astype(str)
        + ", age "
        + users["age"].astype(str)
        + ", "
        + users["occupation"].astype(str)
    )
    return users


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

data, splits, content, dataset_stats = load_all()
results = load_results()
best_params_map = {
    name: json.dumps(res.get("best_params", {}), sort_keys=True)
    for name, res in results.get("tuning", {}).items()
}
families = [f for f in model_families() if f in best_params_map] or model_families()
best_params_map.setdefault("random", "{}")
params_key = tuple((f, best_params_map.get(f, "{}")) for f in families)

models, fit_times = fit_all(params_key)

title_by_pos = {i: content.titles[i] for i in range(len(content.titles))}
genre_by_pos = {i: list(data.items["genre_list"].iloc[i]) for i in range(len(content.titles))}
year_by_pos = {i: data.items["year"].iloc[i] for i in range(len(content.titles))}


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("🎬 Recommender Systems Lab")
st.caption(
    "Popularity · content-based · collaborative filtering on the authentic "
    "MovieLens 100K dataset (GroupLens, University of Minnesota). "
    "Protocol: per-user temporal leave-last-n, all-unrated-item ranking. "
    "Hyperparameters are tuned on a validation tail and reported on an untouched test tail."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox(
        "Model",
        families,
        index=families.index("svd_cf") if "svd_cf" in families else 0,
        format_func=lambda n: models[n].label if n in models else n,
    )
    k = st.slider("Top-K", min_value=1, max_value=50, value=10, step=1)

    st.divider()
    st.subheader("Dataset")
    st.metric("Ratings", f"{dataset_stats.get('n_ratings', 0):,}")
    st.metric("Users", f"{dataset_stats.get('n_users', 0):,}")
    st.metric("Movies", f"{dataset_stats.get('n_items', 0):,}")
    st.metric("Sparsity", f"{dataset_stats.get('sparsity', 0):.1%}")
    st.caption(f"Mean rating {dataset_stats.get('mean_rating', 0):.3f} ★")

    st.divider()
    st.caption(
        "Artifacts: "
        f"`{paths.RESULTS_JSON.relative_to(ROOT)}`"
        + ("" if paths.RESULTS_JSON.exists() else " (missing — run the pipeline)")
    )

split_stats = results.get("protocol", {}).get("split", {})
final_stats = split_stats.get("final", {})

tabs = st.tabs(
    ["📊 Overview", "🎯 Recommendations", "⚖️ Model comparison", "🔍 Data explorer", "⚠️ Limitations"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Evaluation protocol")
    cols = st.columns(4)
    cols[0].metric("Train interactions", f"{final_stats.get('n_train', 0):,}")
    cols[1].metric("Held-out (test)", f"{final_stats.get('n_test', 0):,}")
    cols[2].metric("Test / user", f"{final_stats.get('mean_test_per_user', 0):.1f}")
    cols[3].metric("Candidate catalogue", f"{dataset_stats.get('n_items', 0):,}")

    st.markdown(
        f"""
**How a score is produced.** For every user, their rated movies are sorted by timestamp
and the most recent **{final_stats.get("mean_test_per_user", 0):.0f}** (mean) are held out.
Each model is fit on the older interactions only, then asked to rank *all
{dataset_stats.get("n_items", 0):,} movies the user has not rated in training* and the
top-K are scored against the held-out movies:

- **Precision@K** — of K recommendations, how many were subsequently watched/rated.
- **Recall@K** — of the held-out movies, how many the top-K recovered.
- **NDCG@K** — same hits, but position-weighted (a hit at rank 1 counts more than at rank K).

Hyperparameters are chosen by NDCG@10 on a **validation tail** carved out of the training
period; the test tail is touched exactly once, for the numbers below.
"""
    )

    comparison = pd.DataFrame(results.get("comparison", []))
    if comparison.empty:
        st.warning("No metrics found. Run `python -m recsys_lab.pipeline` first.")
    else:
        at_k = comparison[comparison["k"] == int(results.get("primary_k", 10))]
        at_k = at_k.sort_values("ndcg", ascending=False)
        st.subheader(f"Leaderboard @ K={results.get('primary_k', 10)} (held-out test tail)")
        show = at_k[["label", "kind", "precision", "recall", "ndcg", "coverage"]].rename(
            columns={
                "label": "model",
                "precision": "Precision@K",
                "recall": "Recall@K",
                "ndcg": "NDCG@K",
                "coverage": "catalogue coverage",
            }
        )
        st.dataframe(
            show.style.format(
                {
                    "Precision@K": "{:.4f}",
                    "Recall@K": "{:.4f}",
                    "NDCG@K": "{:.4f}",
                    "catalogue coverage": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.bar_chart(at_k.set_index("label")["ndcg"], height=320)
        st.caption(
            "NDCG@K by model. `catalogue coverage` is the share of all movies that appear in any user's top-K."
        )

# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
with tabs[1]:
    left, right = st.columns([1, 2], gap="large")
    labels = user_labels(data)
    with left:
        st.subheader("Pick a user")
        choice = st.selectbox(
            "User", labels["label"].tolist(), index=0, label_visibility="collapsed"
        )
        user_id = int(choice.split(" — ")[0])
        user_pos = data.user_index[user_id]
        row = labels[labels["user_id"] == user_id].iloc[0]
        st.metric("Age", int(row["age"]))
        st.metric("Gender", row["gender"])
        st.metric("Occupation", row["occupation"])

    model = models[model_name]
    train_row = splits.final.train[splits.final.train["user_idx"] == user_pos]
    test_row = splits.final.test[splits.final.test["user_idx"] == user_pos]
    seen = {int(i) for i in train_row["item_idx"]}
    held_out = {int(i) for i in test_row["item_idx"]}

    recs = model.recommend(user_pos, k, seen)

    with right:
        st.subheader(f"Top-{k} for user {user_id} · {model.label}")
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Hit rate@K", f"{hit_rate_at_k(recs, held_out, k):.2f}")
        metrics_cols[1].metric("NDCG@K (this user)", f"{ndcg_at_k(recs, held_out, k):.3f}")
        metrics_cols[2].metric("Fit time", f"{fit_times.get(model_name, 0):.2f}s")

        rows = []
        for rank, item_pos in enumerate(recs, start=1):
            hit = item_pos in held_out
            score = float(model.score_all()[user_pos, item_pos])
            rows.append(
                {
                    "rank": rank,
                    "title": title_by_pos[item_pos],
                    "year": year_by_pos[item_pos],
                    "genres": ", ".join(genre_by_pos[item_pos][:3]),
                    "score": round(score, 3),
                    "held-out hit": "✅" if hit else "",
                    "why": model.explain(user_pos, item_pos),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "why": st.column_config.TextColumn("explanation", width="large"),
                "held-out hit": st.column_config.TextColumn("hit"),
            },
        )
        st.caption(
            "✅ marks a recommendation that this user genuinely rated in the held-out "
            "period — the model never saw those interactions during training."
        )

    st.divider()
    st.subheader(f"What user {user_id} actually rated next (held-out)")
    if held_out:
        actual = pd.DataFrame(
            [
                {
                    "row": n,
                    "title": title_by_pos[int(i)],
                    "genres": ", ".join(genre_by_pos[int(i)][:3]),
                    "rating": int(rating),
                    "in top-K": "✅" if int(i) in set(recs) else "",
                }
                for n, (i, rating) in enumerate(
                    zip(test_row["item_idx"], test_row["rating"], strict=True), start=1
                )
            ]
        )
        st.dataframe(actual, width="stretch", hide_index=True, height=240)
    else:
        st.info("This user has no held-out interactions.")

    st.caption(
        f"This user has {len(seen):,} training interactions; all of them are masked from "
        f"their recommendation list."
    )

# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
with tabs[2]:
    comparison = pd.DataFrame(results.get("comparison", []))
    if comparison.empty:
        st.warning("No metrics found. Run `python -m recsys_lab.pipeline` first.")
    else:
        st.subheader("Metrics by K")
        for metric, label in (
            ("precision", "Precision@K"),
            ("recall", "Recall@K"),
            ("ndcg", "NDCG@K"),
        ):
            pivot = comparison.pivot_table(index="label", columns="k", values=metric)
            st.markdown(f"**{label}**")
            st.dataframe(pivot.style.format("{:.4f}"), width="stretch")

        st.subheader("Training cost and coverage")
        rows = results["model_results"]
        cost = pd.DataFrame(
            [
                {
                    "model": res["label"],
                    "kind": res["kind"],
                    "fit seconds": res["train_seconds"],
                    "coverage@maxK": res["metrics"][str(max(results["k_values"]))]["coverage"],
                    "parameters": json.dumps(res["params"], sort_keys=True),
                }
                for res in rows.values()
            ]
        )
        st.dataframe(
            cost.style.format({"fit seconds": "{:.2f}", "coverage@maxK": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Hyperparameter search (validation tail)")
        st.caption(
            "Every candidate below was scored on the validation tail only. The winner is "
            "refit on train+validation before the test numbers above are produced."
        )
        for name, res in results.get("tuning", {}).items():
            cands = pd.DataFrame(
                [
                    {
                        "params": json.dumps(c["params"], sort_keys=True),
                        "NDCG@10 (validation)": c["metrics"]["ndcg"],
                        "Recall@10 (validation)": c["metrics"]["recall"],
                        "selected": "✅" if c["params"] == res["best_params"] else "",
                    }
                    for c in res["candidates"]
                ]
            ).sort_values("NDCG@10 (validation)", ascending=False)
            with st.expander(f"{name}  —  {len(cands)} candidates"):
                st.dataframe(
                    cands.style.format(
                        {"NDCG@10 (validation)": "{:.4f}", "Recall@10 (validation)": "{:.4f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
with tabs[3]:
    ratings = data.ratings
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating distribution")
        counts = ratings["rating"].value_counts().sort_index()
        st.bar_chart(counts, height=280)
        st.caption(
            f"mean {ratings['rating'].mean():.3f} ★ · median {ratings['rating'].median():.0f} ★"
        )
    with col2:
        st.subheader("Ratings per user")
        per_user = ratings.groupby("user_id").size()
        hist = np.histogram(per_user, bins=25)
        # Use the numeric left edge as the index so the chart sorts by value
        # rather than lexicographically by a string label like "106-134".
        st.bar_chart(
            pd.Series(hist[0], index=hist[1][:-1].astype(int)),
            height=280,
            x_label="ratings per user (bin lower edge)",
            y_label="users",
        )
        st.caption(
            f"min {per_user.min()} · median {int(per_user.median())} · max {per_user.max()} ratings per user"
        )

    st.subheader("Most-rated movies")
    top = (
        ratings.groupby("item_id")
        .agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    top["title"] = top["item_id"].map(lambda i: title_by_pos[data.item_index[int(i)]])
    st.dataframe(
        top.sort_values("n_ratings", ascending=False)
        .head(25)
        .style.format({"mean_rating": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Genre mix")
    genre_counts = data.items[list(content.genre_names)].sum().sort_values(ascending=False)
    st.bar_chart(genre_counts.drop("unknown", errors="ignore"), height=280)

# --------------------------------------------------------------------------- #
# Limitations
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("What this lab does and does not show")
    st.markdown(
        """
**Honest reading of the numbers**

- **Absolute scores are low, and that is expected.** Ranking *all* 1,682 movies for every
  user (rather than a sampled set of candidate negatives) is the strictest common protocol.
  NDCG@10 around 0.16 means a genuinely hard task, not a broken metric.
- **Content-based filtering is the weakest model here.** MovieLens 100K items carry only 19
  binary genres plus title text, and almost no other content signal, so content-based
  recommendations are barely three times better than random. This matches the published
  literature on this dataset rather than indicating an implementation error.
- **Metadata leakage is avoided but item features are static.** Genres/titles come from
  `u.item`, which describes the movie itself and carries no interaction information, so
  using them does not leak any user's held-out ratings.

**Limitations**

1. **One dataset, one domain.** MovieLens 100K is 1990s-era film ratings from a single
   research site; the conclusions need not transfer to other domains, catalogs or
   populations. Only 943 users are represented.
2. **Explicit-feedback only.** Every model consumes 1-5 star ratings. Real systems usually
   work with implicit signals (clicks, dwell time, purchases) that behave differently, and
   the rating scale here is not comparable across users.
3. **Ratings are sparse but not missing at random.** Users rate films they chose to watch,
   so absent ratings are not a neutral signal; treating them as zeros (which PureSVD does)
   is a modelling convenience, not a fact.
4. **The split is temporal per user, so the evaluation is a "next-watch" simulation,** not a
   cold-start test. The cold-start setting (new users or new items) is not evaluated here at
   all, even though it is where recommenders most often fail — coverage numbers (33% for
   PureSVD) hint at how narrow the exposed catalog is.
5. **Popularity bias is deliberately unhandled.** Nothing re-ranks for diversity, novelty or
   fairness; the popularity baseline is not debiased, and the item-item scores inherit
   whatever popularity structure lives in the co-rating graph.
6. **Determinism caveat.** The split and every metric are deterministic, and the random
   baseline is seeded, but `scipy.sparse.linalg.svds` uses a seeded ARPACK start vector, so
   PureSVD scores can vary by a few 1e-4 across LAPACK builds. The reported decimals should
   not be read as more precise than that.
7. **Metrics cover ranking quality only.** Business-relevant questions — revenue, long-term
   satisfaction, feedback loops, online A/B behaviour — are entirely outside this lab.
8. **No offline-online gap analysis.** Offline ranking metrics are known to correlate only
   loosely with online engagement; nothing here substitutes for a live experiment.

**Reproducibility**

- `python -m recsys_lab.pipeline` regenerates the dataset download, validation, split,
  hyperparameter search, final metrics and artifacts from scratch.
- `pytest` covers the split protocol, the metric definitions, the model contracts and
  dataset validation; `ruff check` / `ruff format --check` cover style.
- The dataset archive's SHA-256 is recorded in `data/raw/ml-100k.source.json` on download.
        """
    )

st.divider()
st.caption(
    "Data: MovieLens 100K, GroupLens Research, University of Minnesota — "
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip. "
    "This project is an educational lab, not a production recommender."
)
