# Recommender Systems Lab

Three families of recommender — **popularity**, **content-based** and **collaborative
filtering** — built and evaluated end-to-end on the **authentic MovieLens 100K** dataset
(GroupLens, University of Minnesota), with a held-out protocol that is defined before any
model is trained, a Streamlit dashboard, and a test suite.

Everything in this repository is reproducible from a single command. No metric in this README
was typed in by hand: `python -m recsys_lab.pipeline` regenerates every number below into
`artifacts/metrics/`.

---

## Headline results (held-out test tail)

Trained on 87,392 interactions, evaluated on 12,608 held-out interactions from all 943 users
(per-user temporal leave-last-n; ranking *all* unseen movies). Hyperparameters were chosen by
NDCG@10 on a **validation tail** carved out of the training period — the test tail was scored
exactly once.

| Model | Selected params | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Precision@20 | Recall@20 | NDCG@20 | Catalogue coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PureSVD (latent factors)** | `{"centered": false, "n_components": 20}` | 0.1540 | 0.0662 | **0.1618** | 0.1368 | 0.1180 | **0.1594** | 0.1142 | 0.1979 | **0.1738** | 0.335 |
| **Item-item CF** | `{"neighbor_k": 50, "shrink": 100.0}` | 0.1120 | 0.0463 | 0.1184 | 0.0902 | 0.0735 | 0.1084 | 0.0731 | 0.1181 | 0.1125 | 0.341 |
| **Popularity (rating count)** | `{"mode": "count"}` | 0.0863 | 0.0358 | 0.0911 | 0.0810 | 0.0638 | 0.0916 | 0.0662 | 0.1025 | 0.0949 | 0.084 |
| **Popularity (Bayesian)** | `{"mode": "bayes", "prior_count": 100.0}` | 0.0632 | 0.0237 | 0.0683 | 0.0554 | 0.0413 | 0.0647 | 0.0452 | 0.0666 | 0.0653 | 0.067 |
| **Content-based** | `{"profile_scheme": "raw", "use_genre": false, "use_title": true}` | 0.0305 | 0.0141 | 0.0334 | 0.0249 | 0.0216 | 0.0312 | 0.0196 | 0.0323 | 0.0316 | 0.438 |
| Random (seeded floor) | `{"seed": 0}` | 0.0078 | 0.0030 | 0.0079 | 0.0089 | 0.0074 | 0.0092 | 0.0089 | 0.0139 | 0.0112 | 1.000 |

`catalogue coverage` is the share of the 1,682 movies that appear in at least one user's top-K.

**How to read this.**

- Collaborative filtering wins. PureSVD reaches **NDCG@10 = 0.1594**, about **17.3×** the
  seeded random floor (0.0092) and **1.7×** the popularity baseline (0.0916).
- Content-based filtering is the weakest model — only **3.4×** random. MovieLens 100K items
  carry almost no content signal: 19 binary genres plus the title text. Once the validation
  search was allowed to choose, it preferred **titles alone** over genres, which is consistent
  with genres being too coarse to separate two films a given user likes. This mirrors published
  results on this dataset and is not an implementation artefact.
- The absolute values look small because the protocol is deliberately harsh: every model must
  rank all 1,682 movies, not a sampled candidate set. `NDCG@10 ≈ 0.16` is a normal, hard-won
  number here.
- Recall rises and precision falls with K, as the definitions require; NDCG@10 sits just under
  NDCG@5 for every model because extra slots add hits at low ranks.

Raw artifacts: [`artifacts/metrics/evaluation.json`](artifacts/metrics/evaluation.json) (full
per-model detail plus all 24 tuning candidates),
[`artifacts/metrics/comparison.csv`](artifacts/metrics/comparison.csv) (flat rows).

---

## Quickstart

Requirements: **Python 3.12** and internet access for the one-time dataset download.
Everything (virtualenv, dataset, caches, artifacts) stays inside the project folder on `D:`.

```bash
cd "D:/Revanth projects/Recommender Systems Lab"

# 1. Environment (uv is used here; `python -m venv .venv` works identically)
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
# equivalently: .venv/Scripts/python.exe -m pip install -e ".[dev]"

# 2. Download MovieLens 100K, validate it, tune, evaluate, write artifacts  (~7 s)
.venv/Scripts/python.exe -m recsys_lab.pipeline

# 3. Tests and lint
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check src app tests scripts
.venv/Scripts/python.exe -m ruff format --check src app tests scripts

# 4. Dashboard
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

If you would rather not install the package, prefix the commands with
`PYTHONPATH=src`. The Streamlit app does this for itself, so it runs from a bare checkout.

Path overrides, if you want the data or artifacts elsewhere: `RECSYS_DATA_DIR`,
`RECSYS_ARTIFACTS_DIR`.

### A readable demo without the UI

```bash
.venv/Scripts/python.exe scripts/demo_recommendations.py 1
```

prints every model's top-5 for user 1 with its explanation, and marks which recommendations
the user genuinely rated in the held-out period.

---

## Project layout

```
src/recsys_lab/
  data.py       download, parse and validate the authentic MovieLens 100K
  split.py      per-user temporal leave-last-n, and the nested three-way split
  metrics.py    Precision@K, Recall@K, NDCG@K, hit rate, catalogue coverage
  ranking.py    top-K selection with seen-item masking
  models.py     popularity, content-based, item-item CF, PureSVD, random floor
  tuning.py     candidate search on the validation tail
  evaluate.py   the harness: fit -> score -> mask -> rank -> average
  pipeline.py   the end-to-end CLI
  paths.py      every path, pinned to the project drive
app/streamlit_app.py       five-tab dashboard
scripts/demo_recommendations.py
tests/                     81 tests
```

---

## The evaluation protocol

**Per-user temporal leave-last-n, all-unrated-item ranking.**

1. Sort each user's ratings by timestamp (ties broken by item id, so it is deterministic).
2. Hold out the most recent `n_u = clip(ceil(0.2 · n_u), 1, 20)` interactions. Everything older
   is training data. Realised: **12,608 held-out** interactions, 13.4 per user on average
   (minimum 4), from 100,000 total.
3. Fit each model on training interactions only.
4. Score every (user, item) pair, mask out items the user already rated in training, take the
   top-K.
5. Compute Precision@K, Recall@K and NDCG@K per user against the held-out items and average
   over users.

Why this is defensible:

- **Forward in time.** A model never sees an interaction later than the ones it is asked to
  rank, which mirrors how a recommender is actually used.
- **No negative sampling.** Every unseen item competes, so the metrics cannot be inflated by
  quietly choosing easy negatives.
- **No test-set tuning.** `three_way_split` reserves the newest 25% of the *training* period as
  a validation tail. All 24 hyperparameter candidates were scored there
  (see the "Hyperparameter search" panel in the dashboard); the winners were refit on
  train+validation and scored on the test tail once.
- **Deterministic.** The split, the metrics and the random baseline are all seeded; re-running
  the pipeline reproduces the same numbers.

Both the R@K and the `n_users` denominators are asserted in the test suite, so a change in the
split cannot silently change what a metric means.

---

## The models

| Model | What it does |
|---|---|
| **Popularity (rating count)** | Ranks items by how many training users rated them. The classic non-personalized baseline. |
| **Popularity (Bayesian)** | IMDb-style weighted rating `(v/(v+m))·R + (m/(v+m))·C` with prior `m` tuned on validation. |
| **Content-based** | TF-IDF over movie titles (sublinear TF, English stop words) and/or 19 genre indicators. A user profile is the relevance-weighted centroid of the items they rated above their own mean; items are scored by cosine similarity. |
| **Item-item CF** | Sarwar et al. (2001). Cosine similarity between item vectors of the user-mean-centered rating matrix, damped by shrinkage `co/(co+shrink)`, capped at the top-`k` neighbours, scored by a similarity-weighted sum of the user's centered ratings. |
| **PureSVD** | Cremonesi et al. (2010). Truncated SVD of the rating matrix; items are scored by the reconstruction `U S Vᵀ`. |
| **Random (seeded floor)** | Uniform random scores. Included so "better than nothing" is a measured claim, not an assertion. |

An earlier version of the item-item model normalised its score by the similarity mass. That
made a single weak neighbour produce a score of `+1.0` and collapsed NDCG@10 to 0.0031 —
statistically indistinguishable from random. The unnormalised weighted sum keeps the magnitude
of the evidence and reaches 0.1084. The lesson is recorded here because the bug was subtle and
the metric caught it; the failing and fixed numbers are in the git history.

---

## Explanations

Every recommendation is explainable in plain language, and the dashboard shows the reason next
to each item:

- **Popularity** — "Bayesian weighted rating 4.20 from 263 ratings (avg 4.45); prior m=100."
- **Content-based** — "'Star Trek: Generations (1994)' shares star, trek with titles this user
  rated above their personal average."
- **Item-item CF** — "Because you rated 'Shawshank Redemption, The (1994)' (5 stars, similarity
  +0.29), 'Amadeus (1984)' (5 stars, similarity +0.23)."
- **PureSVD** — "The 20-dimensional latent factors place this near 'Swingers (1996)',
  'Chasing Amy (1997)', which this user rated highly."

These are faithful descriptions of the model's own arithmetic, not post-hoc rationalisations.

---

## The dashboard

`streamlit run app/streamlit_app.py` — five tabs:

1. **Overview** — protocol summary, leaderboard at K, NDCG bar chart.
2. **Recommendations** — pick any of the 943 users, pick a model and K, see the ranked list
   with scores and explanations, and which of them the user *actually* rated next in the
   held-out period.
3. **Model comparison** — metrics by K, training cost and coverage, and every hyperparameter
   candidate with its validation score and which one was selected.
4. **Data explorer** — rating distribution, ratings per user, most-rated movies, genre mix.
5. **Limitations** — the honest reading, in the app itself.

The app refits the tuned models in-process (about a second in total), so it stays in sync with
the source rather than loading stale pickles.

---

## Limitations

Full detail lives in the dashboard's **Limitations** tab; the short version:

1. **One dataset, one domain.** MovieLens 100K is 1990s-era film ratings from one research
   site, 943 users. Nothing here is guaranteed to transfer.
2. **Explicit feedback only.** The models consume 1–5 star ratings, not the implicit signals
   (clicks, dwell time, purchases) that production systems rely on.
3. **Ratings are not missing at random.** Users rate films they chose to watch, so an absent
   rating is not a neutral zero. Treating it as one — which PureSVD does — is a convenience.
4. **No cold-start evaluation.** The temporal split measures "what to watch next" for known
   users and known items. New users and new items are not evaluated, although coverage numbers
   (33% for PureSVD) hint at how narrow the exposed catalogue becomes.
5. **No popularity or fairness correction.** Nothing re-ranks for diversity, novelty or
   fairness, and item-item scores inherit the popularity structure of the co-rating graph.
6. **Determinism caveat.** The split and metrics are deterministic, but
   `scipy.sparse.linalg.svds` uses an ARPACK start vector, so PureSVD can move in the fourth
   decimal across LAPACK builds. Do not read more precision into the table than that.
7. **Offline metrics only.** No online A/B evidence, and no offline–online gap analysis.
   Offline ranking metrics correlate only loosely with real engagement.

---

## Data source and provenance

- **MovieLens 100K**, GroupLens Research, University of Minnesota:
  <https://files.grouplens.org/datasets/movielens/ml-100k.zip>
- Downloaded once into `data/raw/ml-100k/` (git-ignored). The SHA-256 of the archive is written
  to `data/raw/ml-100k.source.json` at download time.
- `validate_ml100k` asserts the counts published in the dataset README — 100,000 ratings,
  943 users, 1,682 movies, ratings in 1–5, sparsity 0.937, mean 3.5299 — and raises if any
  check fails, so a corrupted or substituted dataset cannot reach the evaluation.
- Genre and title metadata come from `u.item`, which describes the movies themselves and
  contains no interaction data. Using it does not leak any user's held-out ratings.
- Please respect the GroupLens usage terms when reusing the data.

## License

MIT for the code in this repository — see [LICENSE](LICENSE). The MovieLens data is subject to
its own terms from GroupLens.
