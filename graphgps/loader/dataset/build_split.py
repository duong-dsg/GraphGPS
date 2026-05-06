"""
build_split.py  —  Train/val/test split for JSLibs.

TWO MODES  (choose with --mode)
--------------------------------
closed   (default, recommended to start)
    Split GRAPHS of the same lib across train/val/test.
    Every lib appears in all three splits — just different function graphs.
    Model is a standard N-class classifier.
    Use this for pipeline debugging and when your deployment target is a
    fixed known list of libs.

open
    Split by LIB — test libs are never seen during training.
    Requires metric-learning / embedding similarity at inference time,
    NOT a softmax classifier head.
    Use this only after closed-set works and you want generalisation to
    unseen libs.

Directory structure assumed
---------------------------
raw/
  axios@1.7.9/            ← lib key  (lib_name@lib_version)
    rollup@4.46.2/        ← bundler@bundlerver
      graphs/
        a.xml
    webpack@5.95.0/
      graphs/
        e.xml

Output
------
raw/split.json

  closed mode:
    { "axios@1.7.9": { "a.xml": "train", "b.xml": "val", ... }, ... }

  open mode:
    { "axios@1.7.9": "train", "lodash@4.17.21": "test", ... }

Usage
-----
    # closed-set split (recommended first step)
    python build_split.py --raw_dir datasets/JSLibs/raw --mode closed

    # open-set split
    python build_split.py --raw_dir datasets/JSLibs/raw --mode open

    # dry run — print summary, do not write
    python build_split.py --raw_dir datasets/JSLibs/raw --mode closed --dry_run

    # generate the split ONCE before any training
    python graphgps/loader/dataset/build_split.py --raw_dir datasets/JSLibs/raw --mode closed --seed 42
"""

import argparse
import json
import os
import os.path as osp
import random
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_BUNDLER_PREFIXES = ("rollup", "webpack", "vite", "parcel", "esbuild", "browserify")


def _is_lib_dir(name: str, raw_dir: str) -> bool:
    path = osp.join(raw_dir, name)
    if not osp.isdir(path):
        return False
    low = name.lower()
    if any(low.startswith(p) for p in _BUNDLER_PREFIXES):
        return False
    if name in ("node_modules", ".git", "__pycache__"):
        return False
    return True


def _lib_base_name(lib_dir_name: str) -> str:
    """axios@1.7.9 → axios  |  @scope/pkg@1.0 → @scope/pkg"""
    if lib_dir_name.startswith("@"):
        parts = lib_dir_name.split("@")
        return "@" + parts[1]
    return lib_dir_name.split("@")[0]


def _graph_files(graphs_dir: str) -> List[str]:
    return sorted(
        f for f in os.listdir(graphs_dir)
        if (f.endswith((".dot", ".xml"))
            and "Zone.Identifier" not in f
            and not f.startswith("_program"))
    )


def _all_graphs_for_lib(lib_dir: str) -> List[Tuple[str, str]]:
    """
    Returns list of (bundler_ver, fname) for every graph file under lib_dir.
    Skips non-directory entries (bundle.js, build.log, …).
    """
    result = []
    for bundler_ver in sorted(os.listdir(lib_dir)):
        bundler_dir = osp.join(lib_dir, bundler_ver)
        if not osp.isdir(bundler_dir):
            continue
        graphs_dir = osp.join(bundler_dir, "graphs")
        if not osp.isdir(graphs_dir):
            continue
        for fname in _graph_files(graphs_dir):
            result.append((bundler_ver, fname))
    return result


def _count_graphs_for_lib(lib_dir: str) -> int:
    return len(_all_graphs_for_lib(lib_dir))


# ---------------------------------------------------------------------------
# MODE A — closed-set  (split graphs, same lib in all splits)
# ---------------------------------------------------------------------------

def build_split_closed(
    raw_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = 42,
) -> Dict:
    """
    For each lib, shuffle all its graphs and assign them:
        first  train_ratio  → train
        next   val_ratio    → val
        rest                → test

    Output schema:
        {
          "axios@1.7.9": {
            "rollup@4.46.2/graphs/func_001.xml": "train",
            "rollup@4.46.2/graphs/func_002.xml": "val",
            "webpack@5.95.0/graphs/func_003.xml": "test",
            ...
          },
          ...
        }

    Keys inside each lib dict are  "bundler@ver/graphs/fname"
    so they are unambiguous across bundlers.
    """
    rng = random.Random(seed)

    lib_dirs = [
        d for d in sorted(os.listdir(raw_dir))
        if _is_lib_dir(d, raw_dir)
    ]
    if not lib_dirs:
        raise RuntimeError(f"No lib directories found in {raw_dir}")

    split_map: Dict[str, Dict[str, str]] = {}
    totals = {"train": 0, "val": 0, "test": 0}

    for lib in lib_dirs:
        lib_dir = osp.join(raw_dir, lib)
        graphs  = _all_graphs_for_lib(lib_dir)   # [(bundler_ver, fname), ...]

        if not graphs:
            continue

        rng.shuffle(graphs)
        n       = len(graphs)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        # ensure at least 1 in test too if enough graphs exist
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val   = max(1, min(n_val,   n - n_train - 1))

        lib_map: Dict[str, str] = {}
        for i, (bundler_ver, fname) in enumerate(graphs):
            key = f"{bundler_ver}/graphs/{fname}"
            if i < n_train:
                lib_map[key] = "train"
            elif i < n_train + n_val:
                lib_map[key] = "val"
            else:
                lib_map[key] = "test"

        split_map[lib] = lib_map

        # tally
        for sp in ("train", "val", "test"):
            totals[sp] += sum(1 for v in lib_map.values() if v == sp)

    _print_summary("closed", split_map.keys(), totals)
    return split_map


# ---------------------------------------------------------------------------
# MODE B — open-set  (split libs, test libs unseen during training)
# ---------------------------------------------------------------------------

def build_split_open(
    raw_dir: str,
    train_ratio: float  = 0.70,
    val_ratio: float    = 0.15,
    seed: int           = 42,
    group_by_base: bool = True,
) -> Dict[str, str]:
    """
    Assigns entire libs to train/val/test.
    Same lib@ver never appears in more than one split.

    Output schema:
        { "axios@1.7.9": "train", "lodash@4.17.21": "test", ... }
    """
    rng = random.Random(seed)

    lib_dirs = [
        d for d in sorted(os.listdir(raw_dir))
        if _is_lib_dir(d, raw_dir)
    ]
    if not lib_dirs:
        raise RuntimeError(f"No lib directories found in {raw_dir}")

    # group by base name to avoid version leakage
    groups: Dict[str, List[str]] = defaultdict(list)
    for lib in lib_dirs:
        key = _lib_base_name(lib) if group_by_base else lib
        groups[key].append(lib)

    # sort groups by total graph count (descending) for stratification
    group_sizes: List[Tuple[int, str]] = sorted(
        [
            (sum(_count_graphs_for_lib(osp.join(raw_dir, m)) for m in members), base)
            for base, members in groups.items()
        ],
        reverse=True,
    )

    n = len(group_sizes)

    # Guarantee at least 1 group in each split regardless of n or ratios.
    # With only 3 groups: train=1, val=1, test=1.
    # With only 2 groups: train=1, val=1, test=0  (warn user).
    if n < 3:
        print(f"\n⚠️  WARNING: only {n} lib group(s) found — too few for a "
              f"meaningful 3-way split.\n"
              f"   Add more libs to raw/ before training.\n"
              f"   Assigning: train={max(1,n-1)}  val={min(1,n-1)}  test={max(0,n-2)}")

    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))
    # clamp so train + val never exceeds n, leaving at least 1 for test if possible
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val   = max(1, n - n_train - 1) if n > 2 else min(1, n - n_train)
    thresholds = [n_train, n_train + n_val]

    shuffled: List[Tuple[int, str]] = []
    for _, stratum in groupby(group_sizes, key=lambda t: t[0]):
        chunk = list(stratum)
        rng.shuffle(chunk)
        shuffled.extend(chunk)

    split_map: Dict[str, str] = {}
    for i, (_, base) in enumerate(shuffled):
        sp = "train" if i < thresholds[0] else ("val" if i < thresholds[1] else "test")
        for member in groups[base]:
            split_map[member] = sp

    lib_counts   = {"train": 0, "val": 0, "test": 0}
    graph_counts = {"train": 0, "val": 0, "test": 0}
    for lib, sp in split_map.items():
        lib_counts[sp]   += 1
        graph_counts[sp] += _count_graphs_for_lib(osp.join(raw_dir, lib))

    _print_summary("open", split_map.keys(), graph_counts, lib_counts)
    return split_map


# ---------------------------------------------------------------------------
# pretty printer
# ---------------------------------------------------------------------------

def _print_summary(mode, lib_names, graph_counts, lib_counts=None):
    print(f"\nMode: {mode}")
    print(f"{'Split':<8}  {'Libs':>6}  {'Graphs':>8}")
    print("-" * 30)
    total_g = total_l = 0
    for sp in ("train", "val", "test"):
        lc = lib_counts[sp] if lib_counts else "—"
        gc = graph_counts[sp]
        print(f"  {sp:<6}  {str(lc):>6}  {gc:>8}")
        total_g += gc
        if lib_counts:
            total_l += lib_counts[sp]
    print(f"  {'total':<6}  {str(total_l) if lib_counts else '—':>6}  {total_g:>8}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build train/val/test split for JSLibs dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir", default="datasets/JSLibs/raw")
    parser.add_argument("--mode",    default="closed", choices=["closed", "open"],
                        help="closed = same libs in all splits (classifier); "
                             "open   = test libs unseen in training (metric learning)")
    parser.add_argument("--train",   type=float, default=0.70)
    parser.add_argument("--val",     type=float, default=0.15)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--no_group_versions", action="store_true",
                        help="[open mode only] treat each lib@version independently")
    parser.add_argument("--out",     default=None,
                        help="Output path (default: <raw_dir>/split.json)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print summary without writing file")
    args = parser.parse_args()

    if args.mode == "closed":
        result = build_split_closed(
            raw_dir=args.raw_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
        )
    else:
        result = build_split_open(
            raw_dir=args.raw_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
            group_by_base=not args.no_group_versions,
        )

    if args.dry_run:
        print("\n[dry_run] split.json not written.")
        return

    out_path = args.out or osp.join(args.raw_dir, "split.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()