"""
eda.py — Exploratory Data Analysis for JSLibs CPG dataset

Usage
-----
# Graphs in raw/ (original)
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw

# Graphs in a separate data_dir
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw \
              --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2

# Filter to specific bundlers
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw \
              --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
              --bundler rollup@4.46.2 webpack@5.95.0
python scripts/explore/eda.py \
    --raw_dir  datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --lib   async axios lodash express chalk commander react request \
    --bundler rollup@4.46.2 webpack@5.95.0 \
    --save_dir scripts/plots/eda_output/

# Only specific sections
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw --sections overview split sizes

# Save all figures
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw --save_dir scripts/plots/eda_output/

Sections
--------
  overview  — directory tree, lib/bundler counts
  split     — train/val/test distribution
  sizes     — node/edge count histograms and per-lib box plots
  edges     — edge type and edge group distribution
  vocab     — cpg_vocab.json inspection  (always from raw_dir)
  graphs    — sample CPG visualisations (NetworkX spring layout)
"""

import argparse
import json
import os
import os.path as osp
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ── optional imports (graceful degradation) ────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — figures will be skipped.")

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False
    print("[WARN] pandas not found — tables will use plain dicts.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not found — graph visualisation will be skipped.")

try:
    import pydot
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

import xml.etree.ElementTree as ET


# =============================================================================
# Constants
# =============================================================================

EDGE_GROUPS = {
    "AST": 0,         "CONTAINS": 0,
    "CFG": 1,         "DOMINATE": 1,    "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4,        "ARGUMENT": 4,    "PARAMETER_LINK": 4,
    "REF": 5,
}
GROUP_NAMES = {
    0: "Syntax",
    1: "Control Flow",
    2: "Data Flow",
    3: "Control Dep",
    4: "Call / Arg",
    5: "Reference",
}
EDGE_COLORS = {
    0: "#4C72B0",
    1: "#DD8452",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B2",
    5: "#937860",
}

SEP  = "─" * 60
SEP2 = "═" * 60


# =============================================================================
# I/O helpers
# =============================================================================

def _read_dot(path: str):
    if not HAS_PYDOT:
        return None
    try:
        graphs = pydot.graph_from_dot_file(path)
        if not graphs:
            return None
        P = graphs[0]
        G = nx.MultiDiGraph()
        for node in P.get_nodes():
            name = node.get_name()
            if name in ("node", "graph", "edge"):
                continue
            name  = name.strip('"')
            attrs = {k: v.strip('"') for k, v in node.get_attributes().items()}
            G.add_node(name, **attrs)
        for edge in P.get_edges():
            src   = edge.get_source().strip('"')
            dst   = edge.get_destination().strip('"')
            attrs = {k: v.strip('"') for k, v in edge.get_attributes().items()}
            G.add_edge(src, dst, **attrs)
        return G
    except Exception:
        return None


def _read_xml(path: str):
    for reader in (nx.read_graphml, nx.read_gexf):
        try:
            return nx.MultiDiGraph(reader(path))
        except Exception:
            pass
    try:
        G    = nx.MultiDiGraph()
        root = ET.parse(path).getroot()
        for node in root.findall(".//node"):
            nid = node.get("id")
            if nid is None:
                continue
            attrs = dict(node.attrib)
            for d in node.findall(".//data"):
                if d.get("key") and d.text:
                    attrs[d.get("key")] = d.text
            G.add_node(nid, **attrs)
        for edge in root.findall(".//edge"):
            src, dst = edge.get("source"), edge.get("target")
            if src and dst:
                attrs = dict(edge.attrib)
                for d in edge.findall(".//data"):
                    if d.get("key") and d.text:
                        attrs[d.get("key")] = d.text
                G.add_edge(src, dst, **attrs)
        return G
    except Exception:
        return None


def load_graph(path: str):
    if not HAS_NX:
        return None
    if path.endswith(".dot"):
        return _read_dot(path)
    if path.endswith(".xml"):
        return _read_xml(path)
    return None


def _graph_files(graphs_dir: str):
    return sorted(
        f for f in os.listdir(graphs_dir)
        if f.endswith((".dot", ".xml"))
        and "Zone.Identifier" not in f
        and not f.startswith("_program")
    )


def _bundler_matches(bundler_ver: str, bundler_filter: list) -> bool:
    """True when bundler_filter is empty or bundler_ver matches an entry.
    Supports exact ('rollup@4.46.2') and base-name ('rollup') matching."""
    if not bundler_filter:
        return True
    bname = bundler_ver.split("@")[0]
    return bundler_ver in bundler_filter or bname in bundler_filter


def _lib_matches(lib_ver: str, lib_filter: list) -> bool:
    """True when lib_filter is empty or lib_ver matches an entry.
    Supports exact ('axios@1.7.9') and base-name ('axios') matching."""
    if not lib_filter:
        return True
    base = lib_ver.split("@")[0] if not lib_ver.startswith("@") \
           else "@" + lib_ver.split("@")[1]
    return lib_ver in lib_filter or base in lib_filter


_BUNDLER_PREFIXES = ("rollup", "webpack", "vite",
                     "parcel", "esbuild", "browserify")


def _is_lib_dir(name: str, parent: str) -> bool:
    """True if name looks like a lib@ver dir (not a bundler or hidden dir)."""
    if not osp.isdir(osp.join(parent, name)):
        return False
    low = name.lower()
    if any(low.startswith(p) for p in _BUNDLER_PREFIXES):
        return False
    if name in ("node_modules", ".git", "__pycache__",
                "raw", "processed"):
        return False
    return True


def _save_or_show(fig, save_dir: str, fname: str):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = osp.join(save_dir, fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  → saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Section 1 — Overview
# =============================================================================

def section_overview(raw_dir: str, lib_split: dict, save_dir: str, helpers: dict = None, data_dir: str = None, bundler_filter: list = None, lib_filter: list = None):
    print(f"\n{SEP2}")
    print("  SECTION 1 — DIRECTORY OVERVIEW")
    print(SEP2)

    graph_root = data_dir or raw_dir
    bundler_filter = bundler_filter or []
    lib_filter     = lib_filter or []
    rows = []
    for lib_ver in sorted(os.listdir(graph_root)):
        lib_dir = osp.join(graph_root, lib_ver)
        if not _is_lib_dir(lib_ver, graph_root):
            continue
        if not _lib_matches(lib_ver, lib_filter):
            continue
        for bundler_ver in sorted(os.listdir(lib_dir)):
            bundler_dir = osp.join(lib_dir, bundler_ver)
            if not osp.isdir(bundler_dir):
                continue
            if not _bundler_matches(bundler_ver, bundler_filter):
                continue
            graphs_dir = osp.join(bundler_dir, "graphs")
            if not osp.isdir(graphs_dir):
                n_graphs, exts = 0, set()
            else:
                files    = _graph_files(graphs_dir)
                n_graphs = len(files)
                exts     = set(osp.splitext(f)[1] for f in files)

            bundler_name = bundler_ver.split("@")[0]

            # Normalise split value — split.json may store a plain string
            # ("train") or a nested dict ({"split": "train", ...}).
            if helpers and "lib_split_label" in helpers:
                split_val = helpers["lib_split_label"](lib_ver)
            else:
                raw = lib_split.get(lib_ver, "unknown")
                split_val = str(next(iter(raw.values()), "unknown")) \
                            if isinstance(raw, dict) else str(raw)
                split_val = "val" if split_val == "valid" else split_val

            rows.append({
                "lib_ver"    : lib_ver,
                "bundler"    : bundler_name,
                "bundler_ver": bundler_ver,
                "split"      : split_val,
                "n_graphs"   : n_graphs,
                "file_types" : ", ".join(sorted(exts)) or "\u2014",
            })

    n_libs     = len(set(r["lib_ver"]  for r in rows))
    n_bundlers = len(set(r["bundler"]  for r in rows))
    n_combos   = len(rows)
    n_total    = sum(r["n_graphs"] for r in rows)

    print(f"\n  Lib versions      : {n_libs}")
    print(f"  Unique bundlers   : {n_bundlers}  "
          f"({', '.join(sorted(set(r['bundler'] for r in rows)))})")
    print(f"  lib × bundler     : {n_combos}")
    print(f"  Total graph files : {n_total}")

    print(f"\n  {'lib@ver':<30} {'bundler':<12} {'split':<8} {'graphs':>7}  file_types")
    print(f"  {SEP}")
    for r in rows:
        print(f"  {r['lib_ver']:<30} {r['bundler']:<12} {r['split']:<8} "
              f"{r['n_graphs']:>7}  {r['file_types']}")

    return rows


# =============================================================================
# Section 2 — Split distribution
# =============================================================================

def section_split(rows: list, save_dir: str):
    print(f"\n{SEP2}")
    print("  SECTION 2 — SPLIT DISTRIBUTION")
    print(SEP2)

    split_libs   = defaultdict(set)
    split_graphs = defaultdict(int)
    for r in rows:
        split_libs[r["split"]].add(r["lib_ver"])
        split_graphs[r["split"]] += r["n_graphs"]

    print(f"\n  {'split':<10} {'libs':>6} {'graphs':>8}")
    print(f"  {SEP[:30]}")
    for sp in ("train", "val", "test", "unknown"):
        if sp in split_libs:
            print(f"  {sp:<10} {len(split_libs[sp]):>6} {split_graphs[sp]:>8}")
    total_g = sum(split_graphs.values())
    total_l = sum(len(v) for v in split_libs.values())
    print(f"  {'TOTAL':<10} {total_l:>6} {total_g:>8}")

    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    splits    = [s for s in ("train","val","test","unknown") if s in split_libs]
    lib_vals  = [len(split_libs[s])   for s in splits]
    grph_vals = [split_graphs[s]      for s in splits]
    colors    = ["#4C72B0","#DD8452","#55A868","#aaa"][:len(splits)]

    for ax, vals, title in zip(axes,
                                [lib_vals, grph_vals],
                                ["Lib versions per split", "Graphs per split"]):
        wedges, texts, autotexts = ax.pie(
            vals, labels=splits, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(title, fontsize=11, pad=12)

    fig.suptitle("Train / Val / Test Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "02_split_distribution.png")


# =============================================================================
# Section 3 — Graph sizes
# =============================================================================

def section_sizes(raw_dir: str, rows: list, save_dir: str,
                  sample_per_bundler: int = 40, helpers: dict = None,
                  data_dir: str = None, lib_filter: list = None):
    print(f"\n{SEP2}")
    print("  SECTION 3 — GRAPH SIZE DISTRIBUTION")
    print(SEP2)
    print(f"  (sampling up to {sample_per_bundler} graphs per lib×bundler combo)")

    size_rows = []
    lib_filter = lib_filter or []
    for r in rows:
        if not _lib_matches(r["lib_ver"], lib_filter):
            continue
        graph_root = data_dir or raw_dir
        graphs_dir = osp.join(graph_root, r["lib_ver"], r["bundler_ver"], "graphs")
        if not osp.isdir(graphs_dir):
            continue
        fnames = _graph_files(graphs_dir)[:sample_per_bundler]
        for fname in fnames:
            G = load_graph(osp.join(graphs_dir, fname))
            if G is None:
                continue
            # per-graph split (closed mode) or lib-level (open mode)
            if helpers and "graph_split_label" in helpers:
                g_split = helpers["graph_split_label"](
                    r["lib_ver"], r["bundler_ver"], fname)
            else:
                g_split = r["split"]
            size_rows.append({
                "lib_ver": r["lib_ver"],
                "bundler": r["bundler"],
                "split"  : g_split,
                "nodes"  : G.number_of_nodes(),
                "edges"  : G.number_of_edges(),
            })

    if not size_rows:
        print("  [WARN] No graphs could be loaded for size analysis.")
        return size_rows

    nodes = np.array([r["nodes"] for r in size_rows])
    edges = np.array([r["edges"] for r in size_rows])

    print(f"\n  Sampled {len(size_rows)} graphs\n")
    for name, arr in [("nodes", nodes), ("edges", edges)]:
        print(f"  {name}:")
        print(f"    min={arr.min()}  p25={int(np.percentile(arr,25))}  "
              f"median={int(np.median(arr))}  p75={int(np.percentile(arr,75))}  "
              f"p99={int(np.percentile(arr,99))}  max={arr.max()}")

    if not HAS_MPL:
        return size_rows

    # ── histogram ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, arr, col, color in zip(
        axes,
        [nodes, edges],
        ["nodes", "edges"],
        ["#4C72B0", "#DD8452"],
    ):
        clip = int(np.percentile(arr, 99))
        vals = np.clip(arr, 0, clip)
        ax.hist(vals, bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(arr), color="black", linestyle="--",
                   linewidth=1.2, label=f"median = {int(np.median(arr))}")
        ax.axvline(np.mean(arr),   color="red",   linestyle=":",
                   linewidth=1.0, label=f"mean   = {int(np.mean(arr))}")
        ax.set_xlabel(col.capitalize(), fontsize=10)
        ax.set_ylabel("# graphs", fontsize=10)
        ax.set_title(f"{col.capitalize()} per graph  (clipped at p99={clip})",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle("Graph Size Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "03a_size_histogram.png")

    # ── box plot per lib ───────────────────────────────────────────────────────
    libs_uniq = sorted(set(r["lib_ver"] for r in size_rows))
    lib_nodes = [
        [r["nodes"] for r in size_rows if r["lib_ver"] == lib]
        for lib in libs_uniq
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(libs_uniq) * 0.9), 5))
    bp = ax.boxplot(lib_nodes, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"],
                             plt.cm.tab20.colors[:len(libs_uniq)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(libs_uniq) + 1))
    ax.set_xticklabels(libs_uniq, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Node count", fontsize=10)
    ax.set_title("Node count per lib  (box = IQR, whiskers = 1.5×IQR)",
                 fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "03b_nodes_per_lib.png")

    return size_rows


# =============================================================================
# Section 4 — Edge types
# =============================================================================

def section_edges(raw_dir: str, rows: list, save_dir: str,
                  max_graphs: int = 200, data_dir: str = None,
                  lib_filter: list = None):
    print(f"\n{SEP2}")
    print("  SECTION 4 — EDGE TYPE DISTRIBUTION")
    print(SEP2)
    print(f"  (sampling up to {max_graphs} graphs total)")

    edge_counter  = Counter()
    group_counter = Counter()
    sampled       = 0

    lib_filter = lib_filter or []
    for r in rows:
        if sampled >= max_graphs:
            break
        if not _lib_matches(r["lib_ver"], lib_filter):
            continue
        graph_root = data_dir or raw_dir
        graphs_dir = osp.join(graph_root, r["lib_ver"], r["bundler_ver"], "graphs")
        if not osp.isdir(graphs_dir):
            continue
        for fname in _graph_files(graphs_dir)[:10]:
            if sampled >= max_graphs:
                break
            G = load_graph(osp.join(graphs_dir, fname))
            if G is None:
                continue
            for _, _, attr in G.edges(data=True):
                etype = attr.get("label") or attr.get("type") or "AST"
                edge_counter[etype] += 1
                grp = EDGE_GROUPS.get(etype, 0)
                group_counter[GROUP_NAMES.get(grp, "Other")] += 1
            sampled += 1

    total = sum(edge_counter.values())
    print(f"\n  Sampled {sampled} graphs  |  {total:,} edges total\n")
    print(f"  {'edge_type':<22} {'count':>8}  {'%':>6}")
    print(f"  {SEP[:40]}")
    for etype, cnt in edge_counter.most_common():
        print(f"  {etype:<22} {cnt:>8,}  {cnt/total*100:>5.1f}%")

    print(f"\n  {'group':<22} {'count':>8}  {'%':>6}")
    print(f"  {SEP[:40]}")
    for grp, cnt in group_counter.most_common():
        print(f"  {grp:<22} {cnt:>8,}  {cnt/total*100:>5.1f}%")

    if not HAS_MPL or not edge_counter:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw edge types
    top_n   = 15
    items   = edge_counter.most_common(top_n)
    e_names = [i[0] for i in items]
    e_vals  = [i[1] for i in items]
    colors  = [EDGE_COLORS.get(EDGE_GROUPS.get(n, 0), "#aaa") for n in e_names]
    axes[0].barh(e_names[::-1], e_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].set_title(f"Top {top_n} edge types", fontsize=11)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    # grouped
    grp_items  = sorted(group_counter.items(), key=lambda x: -x[1])
    g_names    = [i[0] for i in grp_items]
    g_vals     = [i[1] for i in grp_items]
    grp_colors = [EDGE_COLORS.get(
        next((k for k, v in GROUP_NAMES.items() if v == n), 0), "#aaa")
        for n in g_names]
    axes[1].barh(g_names[::-1], g_vals[::-1], color=grp_colors[::-1])
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].set_title("Edge group distribution", fontsize=11)
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    fig.suptitle("Edge Type Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "04_edge_types.png")


# =============================================================================
# Section 5 — Vocabulary
# =============================================================================

def section_vocab(raw_dir: str, save_dir: str):
    print(f"\n{SEP2}")
    print("  SECTION 5 — NODE LABEL VOCABULARY")
    print(SEP2)

    vocab_path = osp.join(raw_dir, "cpg_vocab.json")
    if not osp.exists(vocab_path):
        print(f"\n  [WARN] {vocab_path} not found.")
        print("  Run:  python -m graphgps.loader.dataset.cpg_vocab "
              f"--raw_dir {raw_dir}")
        return

    with open(vocab_path) as f:
        vocab = json.load(f)

    print(f"\n  Vocab size: {len(vocab)} entries")

    if len(vocab) <= 1:
        print("\n  [WARN] Only UNK in vocab — label extraction likely failed.")
        print("  Run:  python -m graphgps.loader.dataset.cpg_vocab "
              f"--raw_dir {raw_dir} --inspect")
        return

    items = list(vocab.items())[1:]   # skip UNK
    print(f"\n  {'rank':<6} {'label':<35} {'idx':>5}")
    print(f"  {SEP[:50]}")
    for rank, (label, idx) in enumerate(items[:40], start=1):
        print(f"  {rank:<6} {label:<35} {idx:>5}")
    if len(items) > 40:
        print(f"  ... ({len(items) - 40} more entries)")

    if not HAS_MPL or len(items) < 2:
        return

    top = items[:30]
    labels_plot = [x[0][:22] for x in top]
    # frequency is implicitly encoded by position (most_common order)
    freq_rank   = list(range(len(top), 0, -1))

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.barh(labels_plot[::-1], freq_rank[::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, len(top))))
    ax.set_xlabel("Relative frequency rank  (higher = more frequent)", fontsize=10)
    ax.set_title("Top 30 CPG node label types  (sorted by corpus frequency)",
                 fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "05_vocab_top30.png")


# =============================================================================
# Section 6 — Sample graph visualisation
# =============================================================================

def section_graphs(raw_dir: str, rows: list, save_dir: str,
                   n_samples: int = 3, max_nodes: int = 80,
                   helpers: dict = None, data_dir: str = None,
                   lib_filter: list = None):
    print(f"\n{SEP2}")
    print("  SECTION 6 — SAMPLE GRAPH VISUALISATIONS")
    print(SEP2)

    if not HAS_NX or not HAS_MPL:
        print("  [SKIP] networkx and matplotlib both required.")
        return

    lib_filter = lib_filter or []
    shown = 0
    for r in rows:
        if shown >= n_samples:
            break
        if not _lib_matches(r["lib_ver"], lib_filter):
            continue
        graph_root = data_dir or raw_dir
        graphs_dir = osp.join(graph_root, r["lib_ver"], r["bundler_ver"], "graphs")
        if not osp.isdir(graphs_dir):
            continue
        for fname in _graph_files(graphs_dir):
            G = load_graph(osp.join(graphs_dir, fname))
            if G is None or G.number_of_nodes() < 4:
                continue

            if helpers and "graph_split_label" in helpers:
                g_split = helpers["graph_split_label"](
                    r["lib_ver"], r["bundler_ver"], fname)
            else:
                g_split = r["split"]
            title = (f"{r['lib_ver']}  /  {r['bundler_ver']}  /  {fname}"
                     f"  [{g_split}]")
            _visualise_graph(G, title, max_nodes, save_dir,
                             fname=f"06_graph_{shown+1:02d}.png")
            shown += 1
            break

    if shown == 0:
        print("  [WARN] No graphs found to visualise.")


def _visualise_graph(G, title: str, max_nodes: int,
                     save_dir: str, fname: str):
    info = (f"  {G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges")
    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G     = G.subgraph(nodes)
        info += f"  (truncated to {max_nodes} nodes)"
    print(f"\n  {title}")
    print(info)

    pos = nx.spring_layout(G, seed=42,
                            k=1.8 / max(1, G.number_of_nodes() ** 0.5))

    # group edges by type
    edge_groups = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        etype = attr.get("label") or attr.get("type") or "AST"
        edge_groups[EDGE_GROUPS.get(etype, 0)].append((u, v))

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f8f8f8")

    nx.draw_networkx_nodes(
        G, pos, node_size=90, node_color="#AEC6CF", alpha=0.92, ax=ax,
    )
    for grp, edges in edge_groups.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            edge_color=EDGE_COLORS.get(grp, "#999"),
            arrows=True, arrowsize=10,
            width=0.9, alpha=0.65, ax=ax,
        )

    # short node labels
    node_labels = {}
    for n, attr in G.nodes(data=True):
        lbl = attr.get("label", str(n))
        node_labels[n] = lbl[:16] + "…" if len(lbl) > 16 else lbl
    nx.draw_networkx_labels(G, pos, labels=node_labels,
                             font_size=5, ax=ax)

    # legend (only groups present in this graph)
    legend_handles = [
        Patch(color=EDGE_COLORS[g], label=GROUP_NAMES[g])
        for g in sorted(edge_groups.keys())
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.9)

    ax.set_title(f"{title}\n{info.strip()}", fontsize=9, pad=8)
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, save_dir, fname)


# =============================================================================
# Main
# =============================================================================

ALL_SECTIONS = ["overview", "split", "sizes", "edges", "vocab", "graphs"]


def main():
    parser = argparse.ArgumentParser(
        description="EDA for JSLibs CPG dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",   default="datasets/JSLibs/raw",
                        help="Path to raw/ directory "
                             "(split.json and vocab.json always read from here)")
    parser.add_argument("--data_dir",  default=None,
                        help="Directory containing lib@ver/ graph subdirs. "
                             "Defaults to --raw_dir when not set.")
    parser.add_argument("--bundler",   nargs="+", default=[], metavar="BUNDLER",
                        help="Filter to specific bundler versions, e.g. "
                             "rollup@4.46.2 webpack@5.95.0. "
                             "Accepts exact or base names. Default: all.")
    parser.add_argument("--lib",       nargs="+", default=[], metavar="LIB",
                        help="Filter to specific lib versions, e.g. "
                             "axios@1.7.9 lodash chalk@5.3.0. "
                             "Accepts exact (axios@1.7.9) or base (axios) names. "
                             "Default: all libs.")
    parser.add_argument("--sections",  nargs="+", default=ALL_SECTIONS,
                        choices=ALL_SECTIONS, metavar="SECTION",
                        help=f"Sections to run: {ALL_SECTIONS}")
    parser.add_argument("--save_dir",  default="",
                        help="Directory to save figures (empty = show interactively)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of sample graphs to visualise")
    parser.add_argument("--max_nodes", type=int, default=80,
                        help="Max nodes to render per sample graph")
    parser.add_argument("--sample_per_bundler", type=int, default=40,
                        help="Max graphs to load per lib×bundler for size stats")
    parser.add_argument("--edge_sample", type=int, default=200,
                        help="Max graphs to sample for edge type analysis")
    args = parser.parse_args()

    # ── validate paths ─────────────────────────────────────────────────────────
    if not osp.isdir(args.raw_dir):
        sys.exit(f"[ERROR] raw_dir not found: {args.raw_dir}")

    data_dir = args.data_dir or None
    if data_dir and not osp.isdir(data_dir):
        sys.exit(f"[ERROR] data_dir not found: {data_dir}")
    graph_root     = data_dir or args.raw_dir
    bundler_filter = args.bundler or []
    lib_filter     = args.lib or []

    split_json = osp.join(args.raw_dir, "split.json")
    if not osp.exists(split_json):
        sys.exit(f"[ERROR] split.json not found: {split_json}\n"
                 f"  Run build_split.py first.")

    with open(split_json) as f:
        lib_split = json.load(f)

    # Detect split schema:
    #   open   → {"lib@ver": "train"}          (lib-level assignment)
    #   closed → {"lib@ver": {"bund/graphs/f": "train", ...}}  (graph-level)
    first_val  = next(iter(lib_split.values()), None)
    split_mode = "closed" if isinstance(first_val, dict) else "open"
    print(f"  split.json mode : {split_mode}")

    # For EDA, flatten closed-set to lib-level by majority vote
    # (the split a lib contributes most graphs to).
    def _lib_split_label(lib_ver: str) -> str:
        val = lib_split.get(lib_ver, "unknown")
        if isinstance(val, dict):
            counts = Counter(v for v in val.values()
                             if isinstance(v, str))
            if not counts:
                return "unknown"
            sp = counts.most_common(1)[0][0]
        else:
            sp = str(val)
        return "val" if sp == "valid" else sp

    # Graph-level lookup for closed mode (used in section_sizes / section_graphs)
    def _graph_split_label(lib_ver: str, bundler_ver: str, fname: str) -> str:
        val = lib_split.get(lib_ver, {})
        if isinstance(val, dict):
            key = f"{bundler_ver}/graphs/{fname}"
            sp  = val.get(key, "unknown")
        else:
            sp = str(val)
        return "val" if sp == "valid" else sp

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if HAS_MPL:
            matplotlib.use("Agg")     # non-interactive when saving

    print(f"\n{SEP2}")
    print("  JSLibs EDA")
    print(f"  raw_dir   : {osp.abspath(args.raw_dir)}")
    print(f"  graph_root: {osp.abspath(graph_root)}")
    print(f"  bundlers  : {bundler_filter if bundler_filter else 'ALL'}")
    print(f"  libs      : {lib_filter if lib_filter else 'ALL'}")
    print(f"  sections  : {args.sections}")
    print(SEP2)

    # ── run sections ───────────────────────────────────────────────────────────
    rows = []
    helpers = {
        "lib_split_label"  : _lib_split_label,
        "graph_split_label": _graph_split_label,
        "split_mode"       : split_mode,
    }

    if "overview" in args.sections:
        rows = section_overview(args.raw_dir, lib_split, args.save_dir,
                                helpers=helpers, data_dir=data_dir,
                                bundler_filter=bundler_filter,
                                lib_filter=lib_filter)

    if not rows:
        rows = section_overview(args.raw_dir, lib_split, save_dir="",
                                helpers=helpers, data_dir=data_dir,
                                bundler_filter=bundler_filter,
                                lib_filter=lib_filter)

    if "split" in args.sections:
        section_split(rows, args.save_dir)

    if "sizes" in args.sections:
        section_sizes(args.raw_dir, rows, args.save_dir,
                      sample_per_bundler=args.sample_per_bundler,
                      helpers=helpers, data_dir=data_dir,
                      lib_filter=lib_filter)

    if "edges" in args.sections:
        section_edges(args.raw_dir, rows, args.save_dir,
                      max_graphs=args.edge_sample, data_dir=data_dir,
                      lib_filter=lib_filter)

    if "vocab" in args.sections:
        section_vocab(args.raw_dir, args.save_dir)  # vocab always in raw_dir

    if "graphs" in args.sections:
        section_graphs(args.raw_dir, rows, args.save_dir,
                       n_samples=args.n_samples, max_nodes=args.max_nodes,
                       helpers=helpers, data_dir=data_dir,
                       lib_filter=lib_filter)

    print(f"\n{SEP2}")
    print("  EDA complete.")
    if args.save_dir:
        print(f"  Figures saved to: {osp.abspath(args.save_dir)}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()