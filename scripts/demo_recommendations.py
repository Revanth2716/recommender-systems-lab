"""Print a small, human-readable demonstration of every recommender.

Used to sanity-check the dashboard's content and to capture reproducible evidence
for the README.

Run with::

    python scripts/demo_recommendations.py [user_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys_lab import paths  # noqa: E402
from recsys_lab.data import download_ml100k, load_dataset  # noqa: E402
from recsys_lab.evaluate import build_content_from  # noqa: E402
from recsys_lab.models import make_model, model_families  # noqa: E402
from recsys_lab.split import three_way_split  # noqa: E402

K = 5


def main() -> int:
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    paths.ensure_dirs()
    if not paths.ML100K_DIR.joinpath("u.data").exists():
        download_ml100k(paths.RAW_DIR)

    data = load_dataset(paths.ML100K_DIR)
    splits = three_way_split(data)
    content = build_content_from(data)
    train = splits.final.train_matrix()

    tuned: dict[str, dict] = {}
    if paths.RESULTS_JSON.exists():
        payload = json.loads(paths.RESULTS_JSON.read_text(encoding="utf-8"))
        tuned = {
            name: res.get("best_params", {}) for name, res in payload.get("tuning", {}).items()
        }

    user_pos = data.user_index[user_id]
    seen = set(splits.final.seen_train()[user_pos])
    held_out = sorted(splits.final.test_sets()[user_pos])
    titles = content.titles

    profile = data.users[data.users["user_id"] == user_id].iloc[0]
    print(f"User {user_id}: {profile['gender']}, age {profile['age']}, {profile['occupation']}")
    print(f"  training interactions: {len(seen)}   held-out: {len(held_out)}")
    print("  actually watched next: " + ", ".join(titles[i] for i in held_out[:5]))
    print()

    for family in model_families():
        model = make_model(family, tuned.get(family, {}))
        model.fit(train, content)
        recs = model.recommend(user_pos, K, seen)
        # ASCII markers keep the output readable on Windows consoles (cp1252).
        hits = ", ".join(("HIT" if i in held_out else "miss") for i in recs)
        print(f"{model.label}   [{hits}]")
        for rank, item in enumerate(recs, start=1):
            print(f"  {rank}. {titles[item]}")
            print(
                f"     score={model.score_all()[user_pos, item]:.3f}  why: {model.explain(user_pos, item)}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
