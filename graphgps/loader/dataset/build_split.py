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
<data_dir>/                   ← where lib@ver/ graph dirs live
  axios@1.7.9/
    rollup@4.46.2/
      graphs/
        a.xml
    webpack@5.95.0/
      graphs/
        e.xml

<raw_dir>/                    ← where split.json is written
  split.json                  ← OUTPUT

data_dir defaults to raw_dir when not specified (original behaviour).

Output
------
raw/split.json

  closed mode:
    {
      "axios@1.7.9": {
        "rollup@4.46.2/graphs/a.xml": "train",
        "webpack@5.95.0/graphs/e.xml": "val",
        ...
      },
      ...
    }

  open mode:
    { "axios@1.7.9": "train", "lodash@4.17.21": "test", ... }

Usage
-----
    # closed-set split, graphs in raw/ (original)
    python graphgps/loader/dataset/build_split.py --raw_dir datasets/JSLibs/raw --mode closed

    # closed-set split, graphs in a custom path
    python graphgps/loader/dataset/build_split.py \
        --raw_dir  datasets/JSLibs/raw \
        --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
        --mode closed --seed 42

    # open-set split from custom path
    python graphgps/loader/dataset/build_split.py \
        --raw_dir  datasets/JSLibs/raw \
        --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
        --mode open --seed 42

    # dry run — print summary, do not write
    python graphgps/loader/dataset/build_split.py --raw_dir datasets/JSLibs/raw --mode closed --dry_run

    python graphgps/loader/dataset/build_split.py \
    --raw_dir  datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --mode closed --seed 42 \
    --lib   async axios lodash express chalk commander react request rxjs uuid \
    --bundler rollup@4.46.2 webpack@5.95.0
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

_BUNDLER_PREFIXES = ("rollup", "webpack", "vite", "parcel",
                     "esbuild", "browserify")


def _is_lib_dir(name: str, parent: str) -> bool:
    """True if name looks like a lib@ver directory (not a bundler or hidden dir)."""
    if not osp.isdir(osp.join(parent, name)):
        return False
    low = name.lower()
    if any(low.startswith(p) for p in _BUNDLER_PREFIXES):
        return False
    if name in ("node_modules", ".git", "__pycache__", "raw", "processed"):
        return False
    return True


def _lib_matches(lib_ver: str, lib_filter: list) -> bool:
    """True when lib_filter is empty or lib_ver matches an entry.
    Supports exact ('axios@1.7.9') and base-name ('axios') matching."""
    if not lib_filter:
        return True
    base = lib_ver.split("@")[0] if not lib_ver.startswith("@") \
           else "@" + lib_ver.split("@")[1]
    return lib_ver in lib_filter or base in lib_filter


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


def _bundler_matches(bundler_ver: str,
                     bundler_filter: List[str]) -> bool:
    """True if bundler_ver matches any entry in bundler_filter.
    Supports exact match ('rollup@4.46.2') and base-name match ('rollup')."""
    if not bundler_filter:
        return True
    bname = bundler_ver.split("@")[0]
    return bundler_ver in bundler_filter or bname in bundler_filter


def _all_graphs_for_lib(
    lib_dir: str,
    bundler_filter: List[str] = None,
) -> List[Tuple[str, str]]:
    """
    Returns [(bundler_ver, fname), ...] for every graph under lib_dir.
    Key format: bundler@ver/graphs/fname  — matches what process() expects.
    If bundler_filter is set, only matching bundler dirs are included.
    """
    bundler_filter = bundler_filter or []
    result = []
    for bundler_ver in sorted(os.listdir(lib_dir)):
        bundler_dir = osp.join(lib_dir, bundler_ver)
        if not osp.isdir(bundler_dir):
            continue                          # skip bundle.js, build.log, etc.
        if not _bundler_matches(bundler_ver, bundler_filter):
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
    data_dir: str,
    train_ratio: float      = 0.70,
    val_ratio: float        = 0.15,
    seed: int               = 42,
    bundler_filter: List[str] = None,
    lib_filter: List[str]   = None,
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
          },
        }

    Keys inside each lib dict are "bundler@ver/graphs/fname" — these must
    match exactly what jslibs.py builds when walking the same data_dir.
    """
    rng = random.Random(seed)

    lib_dirs = [
        d for d in sorted(os.listdir(data_dir))
        if _is_lib_dir(d, data_dir)
    ]
    if not lib_dirs:
        raise RuntimeError(
            f"No lib directories found in data_dir: {data_dir}\n"
            f"Expected subdirectories like  axios@1.7.9/  with bundler subdirs inside."
        )

    split_map: Dict[str, Dict[str, str]] = {}
    totals = {"train": 0, "val": 0, "test": 0}

    lib_filter = lib_filter or []
    for lib in lib_dirs:
        if not _lib_matches(lib, lib_filter):
            continue
        lib_dir = osp.join(data_dir, lib)
        graphs  = _all_graphs_for_lib(lib_dir, bundler_filter=bundler_filter)

        if not graphs:
            print(f"  [WARN] {lib}: no graph files found — skipped")
            continue

        rng.shuffle(graphs)
        n       = len(graphs)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))

        # ensure at least 1 in each split when there are enough graphs
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val   = max(1, min(n_val,   n - n_train - 1))

        lib_map: Dict[str, str] = {}
        for i, (bundler_ver, fname) in enumerate(graphs):
            key = f"{bundler_ver}/graphs/{fname}"   # ← exact key process() looks up
            if i < n_train:
                lib_map[key] = "train"
            elif i < n_train + n_val:
                lib_map[key] = "val"
            else:
                lib_map[key] = "test"

        split_map[lib] = lib_map

        for sp in ("train", "val", "test"):
            totals[sp] += sum(1 for v in lib_map.values() if v == sp)

    _print_summary("closed", split_map.keys(), totals)
    return split_map


# ---------------------------------------------------------------------------
# MODE B — open-set  (split libs, test libs unseen during training)
# ---------------------------------------------------------------------------

def build_split_open(
    data_dir: str,
    train_ratio: float      = 0.70,
    val_ratio: float        = 0.15,
    seed: int               = 42,
    group_by_base: bool     = True,
    bundler_filter: List[str] = None,
    lib_filter: List[str]   = None,
) -> Dict[str, str]:
    """
    Assigns entire libs to train/val/test.
    Same lib@ver never appears in more than one split.

    Output schema:
        { "axios@1.7.9": "train", "lodash@4.17.21": "test", ... }
    """
    rng = random.Random(seed)

    lib_filter = lib_filter or []
    lib_dirs = [
        d for d in sorted(os.listdir(data_dir))
        if _is_lib_dir(d, data_dir) and _lib_matches(d, lib_filter)
    ]
    if not lib_dirs:
        raise RuntimeError(
            f"No lib directories found in data_dir: {data_dir}"
            + (f" matching lib_filter={lib_filter}" if lib_filter else "")
        )

    # group by base name to avoid version leakage
    groups: Dict[str, List[str]] = defaultdict(list)
    for lib in lib_dirs:
        key = _lib_base_name(lib) if group_by_base else lib
        groups[key].append(lib)

    # sort by total graph count (descending) for stratification
    group_sizes: List[Tuple[int, str]] = sorted(
        [
            (
                sum(
                    len(_all_graphs_for_lib(osp.join(data_dir, m),
                                           bundler_filter=bundler_filter))
                    for m in members
                ),
                base,
            )
            for base, members in groups.items()
        ],
        reverse=True,
    )

    n = len(group_sizes)
    if n < 3:
        print(
            f"\n⚠️  WARNING: only {n} lib group(s) found — too few for a "
            f"meaningful 3-way split.\n"
            f"   Add more libs before training.\n"
            f"   Assigning: train={max(1,n-1)}  val={min(1,n-1)}"
            f"  test={max(0,n-2)}"
        )

    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))
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
        sp = ("train" if i < thresholds[0]
              else "val" if i < thresholds[1]
              else "test")
        for member in groups[base]:
            split_map[member] = sp

    lib_counts   = {"train": 0, "val": 0, "test": 0}
    graph_counts = {"train": 0, "val": 0, "test": 0}
    for lib, sp in split_map.items():
        lib_counts[sp]   += 1
        graph_counts[sp] += len(_all_graphs_for_lib(
            osp.join(data_dir, lib), bundler_filter=bundler_filter))

    _print_summary("open", split_map.keys(), graph_counts, lib_counts)
    return split_map


# ---------------------------------------------------------------------------
# pretty printer
# ---------------------------------------------------------------------------

def _print_summary(mode, lib_names, graph_counts, lib_counts=None):
    print(f"\nMode  : {mode}")
    print(f"Libs  : {len(list(lib_names))}")
    print(f"{'Split':<8}  {'Libs':>6}  {'Graphs':>8}")
    print("─" * 30)
    total_g = total_l = 0
    for sp in ("train", "val", "test"):
        lc = lib_counts[sp] if lib_counts else "—"
        gc = graph_counts.get(sp, 0)
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
    parser.add_argument(
        "--raw_dir", default="datasets/JSLibs/raw",
        help="Directory where split.json will be written "
             "(also used as data_dir when --data_dir is not set)",
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Directory containing lib@ver/ graph subdirectories. "
             "Defaults to --raw_dir when not set.",
    )
    parser.add_argument(
        "--mode", default="closed", choices=["closed", "open"],
        help="closed = same libs in all splits (classifier); "
             "open   = test libs unseen in training (metric learning)",
    )
    parser.add_argument("--train",   type=float, default=0.70)
    parser.add_argument("--val",     type=float, default=0.15)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument(
        "--no_group_versions", action="store_true",
        help="[open mode only] treat each lib@version independently "
             "(may allow version leakage — not recommended)",
    )
    parser.add_argument(
        "--lib", nargs="+", default=[], metavar="LIB",
        help="Lib versions to include. Accepts exact names "
             "('axios@1.7.9') or base names ('axios'). "
             "Multiple values allowed. Default: all libs.",
    )
    parser.add_argument(
        "--bundler", nargs="+", default=[], metavar="BUNDLER",
        help="Bundler versions to include. Accepts exact names "
             "('rollup@4.46.2') or base names ('rollup'). "
             "Multiple values allowed. Default: all bundlers.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path for split.json (default: <raw_dir>/split.json)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print summary without writing file",
    )
    args = parser.parse_args()

    # resolve data_dir — default to raw_dir (original behaviour)
    data_dir = args.data_dir or args.raw_dir
    if not osp.isdir(data_dir):
        raise SystemExit(f"[ERROR] data_dir not found: {data_dir}")
    if not osp.isdir(args.raw_dir):
        os.makedirs(args.raw_dir, exist_ok=True)
        print(f"Created raw_dir: {args.raw_dir}")

    print(f"raw_dir  : {osp.abspath(args.raw_dir)}")
    print(f"data_dir : {osp.abspath(data_dir)}")
    print(f"mode     : {args.mode}  |  seed={args.seed}  "
          f"|  train={args.train}  val={args.val}")
    if args.lib:
        print(f"libs     : {args.lib}")
    else:
        print("libs     : ALL (no filter)")
    if args.bundler:
        print(f"bundlers : {args.bundler}")
    else:
        print("bundlers : ALL (no filter)")

    if args.mode == "closed":
        result = build_split_closed(
            data_dir       = data_dir,
            train_ratio    = args.train,
            val_ratio      = args.val,
            seed           = args.seed,
            bundler_filter = args.bundler or None,
            lib_filter     = args.lib or None,
        )
    else:
        result = build_split_open(
            data_dir       = data_dir,
            train_ratio    = args.train,
            val_ratio      = args.val,
            seed           = args.seed,
            group_by_base  = not args.no_group_versions,
            bundler_filter = args.bundler or None,
            lib_filter     = args.lib or None,
        )

    if args.dry_run:
        print("\n[dry_run] split.json not written.")
        return

    out_path = args.out or osp.join(args.raw_dir, "split.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWrote → {out_path}  ({len(result)} libs)")


if __name__ == "__main__":
    main()
