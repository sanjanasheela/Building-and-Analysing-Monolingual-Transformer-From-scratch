"""
Build reproducible train / validation / test splits for Nepali reasoning.

Validation and test use held-out names and held-out attribute families so
entity strings and relation patterns cannot leak from train.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from template import (
    CATEGORIES,
    HELD_OUT_CHAIN_ATTRS,
    HELD_OUT_NAMES,
    RANDOM_SEED,
    TRAIN_CHAIN_ATTRS,
    TRAIN_NAMES,
    generate_single_sample,
)

_FINETUNE_DIR = Path(__file__).resolve().parent
DATA_DIR = _FINETUNE_DIR / "data"

# Nested train sizes for the scaling experiment (100 → 10k).
TARGET_TRAIN = 10_000
TRAIN_SIZES = [
    100, 250, 500, 1000, 1500, 2000, 2500, 3000,
    4000, 5000, 6000, 7500, 8000, 10_000,
]
N_PER_CATEGORY_VAL = 40
N_PER_CATEGORY_TEST = 40


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stats(rows: list[dict]) -> dict:
    cats = Counter(r["category"] for r in rows)
    hops = Counter(r["hop_count"] for r in rows)
    return {
        "n": len(rows),
        "by_category": dict(cats),
        "by_hop": {str(k): v for k, v in sorted(hops.items())},
    }


def create_splits(data_dir: Path | None = None, seed: int = RANDOM_SEED) -> dict:
    """
    Generate the master splits and nested train subsets.

    Returns a dict of output paths and dataset statistics.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    random.seed(seed)

    train_pool: list[dict] = []
    while len(train_pool) < TARGET_TRAIN:
        for cat in CATEGORIES:
            if len(train_pool) >= TARGET_TRAIN:
                break
            train_pool.append(generate_single_sample(cat, pool="train"))
    rng.shuffle(train_pool)

    val_set: list[dict] = []
    test_set: list[dict] = []
    for cat in CATEGORIES:
        for _ in range(N_PER_CATEGORY_VAL):
            val_set.append(generate_single_sample(cat, pool="held_out"))
        for _ in range(N_PER_CATEGORY_TEST):
            test_set.append(generate_single_sample(cat, pool="held_out"))
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    val_path = data_dir / "val_fixed.jsonl"
    test_path = data_dir / "test_fixed.jsonl"
    train_full_path = data_dir / "train_full.jsonl"
    _write_jsonl(val_path, val_set)
    _write_jsonl(test_path, test_set)
    _write_jsonl(train_full_path, train_pool)

    subset_paths = {}
    for size in TRAIN_SIZES:
        actual = min(size, len(train_pool))
        subset = train_pool[:actual]
        path = data_dir / f"train_samples_{actual}.jsonl"
        _write_jsonl(path, subset)
        subset_paths[actual] = str(path)
        print(
            f"[Splitter] {path.name}: {len(subset)} train | "
            f"val={len(val_set)} | test={len(test_set)}"
        )

    summary = {
        "seed": seed,
        "leakage_controls": {
            "train_names": TRAIN_NAMES,
            "held_out_names": HELD_OUT_NAMES,
            "train_attributes": list(TRAIN_CHAIN_ATTRS),
            "held_out_attributes": list(HELD_OUT_CHAIN_ATTRS),
            "note": (
                "Val/test examples use disjoint person names and disjoint "
                "chain attributes (points/temperature/distance) versus train "
                "(height/age/money/weight)."
            ),
        },
        "template_categories": CATEGORIES,
        "train": _stats(train_pool),
        "val": _stats(val_set),
        "test": _stats(test_set),
        "paths": {
            "train_full": str(train_full_path),
            "val": str(val_path),
            "test": str(test_path),
            "train_subsets": subset_paths,
        },
    }
    stats_path = data_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Splitter] Wrote stats → {stats_path}")
    return summary


if __name__ == "__main__":
    create_splits()
