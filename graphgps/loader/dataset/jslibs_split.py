"""
build_split.py  —  Best-practice train/val/test split for JSLibs.

Design decisions
----------------
1. Split at LIBRARY level, not graph level.
   Rationale: If any graph from lib X appears in train, ALL graphs from lib X
   must stay in train.  Mixing graphs of the same lib across splits leaks
   the class signal directly.

2. Version-aware grouping.
   If your raw/ directory has e.g.  axios@0.27  and  axios@0.28,
   treat them as ONE group and keep them in the same bucket.
   Set GROUP_BY_BASE_NAME=True (default) to enable this.

3. Stratified assignment by graph count.
   Sort lib groups by their total graph count, then assign round-robin
   (largest first) so train/val/test all have similar size distributions.
   This prevents "lodash has 2000 graphs and is only in test" scenarios.

4. Bundler is NOT a split axis.
   Both rollup and webpack graphs of the same lib go into the same bucket.
   The model should generalise across bundlers — keep them together.

5. Reproducible: fixed SEED.

Output: raw/split.json   (lib_name → "train" | "val" | "test")

Usage
-----
    python build_split.py --raw_dir datasets/JSLibs/raw \
                          --train 0.70 --val 0.15 --test 0.15 \
                          --seed 42
"""

import argparse
import json
import os
import os.path as osp
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _base_name(lib: str) -> str:
    """
    Strip semver suffix so  axios@0.27.2  →  axios.
    Works with both  lib@version  and  lib-version  naming conventions.
    """
    return re.split(r'[@\-]\d', lib)[0]


def _count_graphs(lib_dir: str) -> int:
    """Count .dot / .xml graph files under all bundler sub-dirs."""
    total = 0
    for bundler in os.listdir(lib_dir):
        graphs_dir = osp.join(lib_dir, bundler, 'graphs')
        if not osp.isdir(graphs_dir):
            continue
        for fname in os.listdir(graphs_dir):
            if fname.endswith(('.dot', '.xml')) and 'Zone.Identifier' not in fname:
                total += 1
    return total


# ---------------------------------------------------------------------------
# core split logic
# ---------------------------------------------------------------------------

def build_split(
    raw_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    group_by_base: bool = True,
) -> Dict[str, str]:
    """
    Returns  {lib_dir_name: "train" | "val" | "test"}
    for every lib sub-directory found in raw_dir.
    """
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-9

    # ---- 1. discover all lib directories ----
    libs: List[str] = [
        d for d in os.listdir(raw_dir)
        if osp.isdir(osp.join(raw_dir, d))
    ]

    if not libs:
        raise RuntimeError(f"No lib sub-directories found in {raw_dir}")

    # ---- 2. group by base name (version-aware) ----
    # groups: base_name → list of lib dir names
    groups: Dict[str, List[str]] = defaultdict(list)
    for lib in libs:
        key = _base_name(lib) if group_by_base else lib
        groups[key].append(lib)

    # ---- 3. compute graph count per group (for stratification) ----
    group_sizes: List[Tuple[int, str]] = []
    for base, members in groups.items():
        total = sum(
            _count_graphs(osp.join(raw_dir, m))
            for m in members
        )
        group_sizes.append((total, base))

    # sort descending so we distribute large libs first
    group_sizes.sort(reverse=True)

    print(f"\nFound {len(libs)} libs in {len(groups)} groups.")
    print("Top 10 groups by graph count:")
    for cnt, base in group_sizes[:10]:
        print(f"  {base:30s}  {cnt:5d} graphs")

    # ---- 4. stratified round-robin assignment ----
    rng = random.Random(seed)
    # shuffle groups of equal size for randomness within strata
    from itertools import groupby
    shuffled: List[Tuple[int, str]] = []
    for _, grp in groupby(group_sizes, key=lambda t: t[0]):
        chunk = list(grp)
        rng.shuffle(chunk)
        shuffled.extend(chunk)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val   = round(n * val_ratio)
    # n_test  = n - n_train - n_val  (remainder)

    split_map: Dict[str, str] = {}   # lib_dir_name → split

    for i, (_, base) in enumerate(shuffled):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for member in groups[base]:
            split_map[member] = split

    # ---- 5. report ----
    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        counts[s] += 1
    print(f"\nLib-level split:  train={counts['train']}  val={counts['val']}  test={counts['test']}")

    graph_counts = {"train": 0, "val": 0, "test": 0}
    for lib, split in split_map.items():
        graph_counts[split] += _count_graphs(osp.join(raw_dir, lib))
    print(f"Graph-level split: train={graph_counts['train']}  val={graph_counts['val']}  test={graph_counts['test']}")

    return split_map


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build train/val/test split for JSLibs")
    parser.add_argument("--raw_dir",  default="datasets/JSLibs/raw")
    parser.add_argument("--train",    type=float, default=0.70)
    parser.add_argument("--val",      type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--no_group_versions", action="store_true",
                        help="Treat each lib@version as a separate group")
    parser.add_argument("--out",      default=None,
                        help="Output path (default: raw_dir/split.json)")
    args = parser.parse_args()

    split_map = build_split(
        raw_dir=args.raw_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
        group_by_base=not args.no_group_versions,
    )

    out_path = args.out or osp.join(args.raw_dir, "split.json")
    with open(out_path, "w") as f:
        json.dump(split_map, f, indent=2, sort_keys=True)
    print(f"\nWrote split to: {out_path}")


if __name__ == "__main__":
    main()