"""Download, load and validate the authentic MovieLens 100K dataset.

Source: https://files.grouplens.org/datasets/movielens/ml-100k.zip
The dataset is distributed by GroupLens (University of Minnesota) and was
collected by the GroupLens research project. We download the canonical archive
and validate it against the counts published in the dataset README, so the
project never silently substitutes a different or synthetic dataset.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_ARCHIVE = "ml-100k.zip"

# Counts published in ml-100k/README (GroupLens).
EXPECTED_N_RATINGS = 100_000
EXPECTED_N_USERS = 943
EXPECTED_N_ITEMS = 1_682
EXPECTED_N_GENRES = 19

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_ml100k(data_dir: Path, *, force: bool = False) -> Path:
    """Download and extract MovieLens 100K into ``data_dir/ml-100k``.

    Returns the directory containing ``u.data``, ``u.item``, ``u.user``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "ml-100k"
    marker = data_dir / "ml-100k.source.json"
    archive = data_dir / ML100K_ARCHIVE

    if raw_dir.joinpath("u.data").exists() and marker.exists() and not force:
        return raw_dir

    import requests

    if force or not archive.exists():
        resp = requests.get(ML100K_URL, timeout=120, stream=True)
        resp.raise_for_status()
        with archive.open("wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 16):
                fh.write(block)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)

    if not raw_dir.joinpath("u.data").exists():
        raise FileNotFoundError(f"u.data not found after extracting {archive}")

    marker.write_text(
        json.dumps(
            {
                "url": ML100K_URL,
                "archive": archive.name,
                "sha256": sha256_of(archive),
                "bytes": archive.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_dir


def _load_genre_vectors(item_path: Path) -> pd.DataFrame:
    rows = []
    with item_path.open(encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            item_id = int(parts[0])
            rows.append((item_id, *[int(v) for v in parts[5 : 5 + EXPECTED_N_GENRES]]))
    cols = ["item_id", *GENRES]
    return pd.DataFrame(rows, columns=cols)


def load_items(raw_dir: Path) -> pd.DataFrame:
    """Load ``u.item`` movie metadata with a parsed genre vector."""
    raw_dir = Path(raw_dir)
    meta = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"],
        dtype={"item_id": "int64", "title": "string"},
    )
    meta = meta.drop(columns=["video_release_date"])
    genres = _load_genre_vectors(raw_dir / "u.item")
    items = meta.merge(genres, on="item_id", how="inner")
    items["year"] = items["release_date"].str.extract(r"(\d{4})", expand=False).astype("Float64")
    items["genre_list"] = items[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1], axis=1
    )
    return items.sort_values("item_id").reset_index(drop=True)


def load_users(raw_dir: Path) -> pd.DataFrame:
    """Load ``u.user`` demographic metadata."""
    return pd.read_csv(
        Path(raw_dir) / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip"],
        dtype={"user_id": "int64", "occupation": "string", "zip": "string"},
    )


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Load ``u.data`` ratings."""
    return pd.read_csv(
        Path(raw_dir) / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "item_id": "int64",
            "rating": "int8",
            "timestamp": "int64",
        },
    )


@dataclass
class MovieLensData:
    """MovieLens 100K with stable positional indices for users and items."""

    ratings: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    genre_matrix: np.ndarray
    user_ids: np.ndarray = field(repr=False)
    item_ids: np.ndarray = field(repr=False)
    user_index: dict[int, int] = field(repr=False)
    item_index: dict[int, int] = field(repr=False)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def id_to_title(self) -> dict[int, str]:
        return dict(zip(self.items["item_id"], self.items["title"], strict=True))

    def title_for(self, item_id: int) -> str:
        """Return the movie title for a raw MovieLens item id."""
        row = self.items.loc[self.items["item_id"] == item_id, "title"]
        return str(row.iloc[0]) if len(row) else ""


def load_dataset(raw_dir: Path) -> MovieLensData:
    """Load and index MovieLens 100K from an extracted ``ml-100k`` directory."""
    raw_dir = Path(raw_dir)
    ratings = load_ratings(raw_dir)
    items = load_items(raw_dir)
    users = load_users(raw_dir)

    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(items["item_id"].unique())
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(m): i for i, m in enumerate(item_ids)}

    ratings = ratings.assign(
        user_idx=ratings["user_id"].map(user_index).astype("int32"),
        item_idx=ratings["item_id"].map(item_index).astype("int32"),
    )
    genre_matrix = items[GENRES].to_numpy(dtype=np.float32)

    return MovieLensData(
        ratings=ratings,
        items=items,
        users=users,
        genre_matrix=genre_matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
    )


def validate_ml100k(data: MovieLensData) -> dict[str, object]:
    """Validate the loaded data against the counts published by GroupLens.

    Raises AssertionError on any mismatch so a corrupted or substituted dataset
    cannot pass into the evaluation pipeline.
    """
    checks: dict[str, tuple[object, object]] = {
        "n_ratings": (len(data.ratings), EXPECTED_N_RATINGS),
        "n_users": (data.n_users, EXPECTED_N_USERS),
        "n_items": (data.n_items, EXPECTED_N_ITEMS),
        "min_rating": (int(data.ratings["rating"].min()), 1),
        "max_rating": (int(data.ratings["rating"].max()), 5),
    }
    failures = {k: v for k, v in checks.items() if v[0] != v[1]}

    # Every item must appear in u.item and every rating must reference a known id.
    unknown_items = set(data.ratings["item_id"]) - set(data.items["item_id"])
    if unknown_items:
        failures["unknown_item_ids"] = (len(unknown_items), 0)

    if failures:
        raise AssertionError(f"MovieLens 100K validation failed: {failures}")

    sparsity = 1.0 - len(data.ratings) / (data.n_users * data.n_items)
    return {
        **{k: v[0] for k, v in checks.items()},
        "n_genres": len(GENRES),
        "sparsity": round(sparsity, 4),
        "mean_rating": round(float(data.ratings["rating"].mean()), 4),
    }
