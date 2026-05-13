"""
eda.py — Exploratory Data Analysis for JSLibs CPG dataset

Usage
-----
# Individual function graphs (default — excludes _program files)
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2

# Entire whole-program CPGs (_program.xml / _program.dot only)
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --load_type entire

# Filter libs / bundlers
python scripts/explore/eda.py \
    --raw_dir  datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --lib   async axios lodash express chalk commander react request rxjs uuid \
    --bundler rollup@4.46.2 webpack@5.95.0 \
    --load_type entire \
    --save_dir scripts/plots/eda_output/ --sections edges

# Only specific sections
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw --sections overview split sizes

# Save all figures
python scripts/explore/eda.py --raw_dir datasets/JSLibs/raw \
    --save_dir scripts/plots/eda_output/

Sections
--------
  overview  — directory tree, lib/bundler counts
  split     — train/val/test distribution
  sizes     — node/edge count histograms and per-lib box plots
  edges     — edge type and edge group distribution
  vocab     — cpg_vocab.json inspection  (always from raw_dir)
  graphs    — sample CPG visualisations (NetworkX spring layout)

Load types
----------
  individual  — per-function graph files (excludes _program.*)  [default]
  entire      — whole-program CPG only  (_program.xml / _program.dot)
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

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
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

# Joern GraphML: node type is encoded in the upper 32 bits of the integer ID.
JOERN_ID_THRESHOLD = 1 << 32
JOERN_NODE_TYPE_MAP = {
    0: "UNKNOWN", 1: "METHOD", 2: "METHOD_PARAMETER_IN",
    3: "METHOD_PARAMETER_OUT", 4: "METHOD_RETURN", 5: "BLOCK",
    6: "CALL", 7: "IDENTIFIER", 8: "FIELD_IDENTIFIER", 9: "LITERAL",
    10: "LOCAL", 11: "MEMBER", 12: "MODIFIER", 13: "TYPE_DECL",
    14: "TYPE_REF", 15: "RETURN", 16: "CONTROL_STRUCTURE",
    17: "JUMP_TARGET", 25: "COMMENT", 33: "METHOD_REF",
}

LOAD_TYPE_INDIVIDUAL = "individual"
LOAD_TYPE_ENTIRE     = "entire"
LOAD_TYPES           = [LOAD_TYPE_INDIVIDUAL, LOAD_TYPE_ENTIRE]

SEP  = "─" * 60
SEP2 = "═" * 60


# =============================================================================
# Joern helpers
# =============================================================================

def _safe_int(s) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def _joern_node_label(node_id: str) -> str:
    """Decode a Joern node ID → CPG node type name."""
    nid = _safe_int(node_id)
    if nid >= JOERN_ID_THRESHOLD:
        return JOERN_NODE_TYPE_MAP.get(nid >> 32, f"UNKNOWN_{nid >> 32}")
    return ""


def _get_node_label(node_id: str, attr: dict) -> str:
    """
    Return the semantic label for a node.
    Handles both standard CPG (text label attr) and Joern (ID-encoded type).
    """
    text = attr.get("label", "").strip()
    if text:
        return text
    joern = _joern_node_label(node_id)
    if joern:
        return joern
    return "UNK"


def _is_joern_graph(G) -> bool:
    """Heuristic: check first 20 node IDs for Joern encoding."""
    if G is None:
        return False
    sample = list(G.nodes())[:20]
    if not sample:
        return False
    count = sum(1 for n in sample if _safe_int(n) >= JOERN_ID_THRESHOLD)
    return count >= max(1, len(sample) // 2)


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


def _graph_files_individual(graphs_dir: str):
    """
    List per-function graph files.
    Excludes _program.* files and Windows zone-identifier artifacts.
    """
    return sorted(
        f for f in os.listdir(graphs_dir)
        if f.endswith((".dot", ".xml"))
        and "Zone.Identifier" not in f
        and not f.startswith("_program")
    )


def _graph_files_entire(graphs_dir: str):
    """
    List only the whole-program CPG file (_program.xml or _program.dot).
    Returns a list of 0 or 1 filenames.
    """
    for ext in (".xml", ".dot"):
        fname = f"_program{ext}"
        if osp.isfile(osp.join(graphs_dir, fname)):
            return [fname]
    return []


def get_graph_files(graphs_dir: str, load_type: str):
    """
    Dispatch to the correct file-listing function based on load_type.

      individual → per-function files (excludes _program.*)
      entire     → only _program.xml / _program.dot
    """
    if load_type == LOAD_TYPE_ENTIRE:
        return _graph_files_entire(graphs_dir)
    return _graph_files_individual(graphs_dir)


def _bundler_matches(bundler_ver: str, bundler_filter: list) -> bool:
    if not bundler_filter:
        return True
    bname = bundler_ver.split("@")[0]
    return bundler_ver in bundler_filter or bname in bundler_filter


def _lib_matches(lib_ver: str, lib_filter: list) -> bool:
    if not lib_filter:
        return True
    base = (lib_ver.split("@")[0] if not lib_ver.startswith("@")
            else "@" + lib_ver.split("@")[1])
    return lib_ver in lib_filter or base in lib_filter


_BUNDLER_PREFIXES = ("rollup", "webpack", "vite",
                     "parcel", "esbuild", "browserify")


def _is_lib_dir(name: str, parent: str) -> bool:
    if not osp.isdir(osp.join(parent, name)):
        return False
    low = name.lower()
    if any(low.startswith(p) for p in _BUNDLER_PREFIXES):
        return False
    return name not in ("node_modules", ".git", "__pycache__",
                        "raw", "processed")


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

def section_overview(
    raw_dir: str,
    lib_split: dict,
    save_dir: str,
    helpers: dict = None,
    data_dir: str = None,
    bundler_filter: list = None,
    lib_filter: list = None,
    load_type: str = LOAD_TYPE_INDIVIDUAL,
):
    print(f"\n{SEP2}")
    print(f"  SECTION 1 — DIRECTORY OVERVIEW  [{load_type}]")
    print(SEP2)

    graph_root     = data_dir or raw_dir
    bundler_filter = bundler_filter or []
    lib_filter     = lib_filter     or []
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
                files, n_graphs, exts = [], 0, set()
            else:
                files    = get_graph_files(graphs_dir, load_type)
                n_graphs = len(files)
                exts     = set(osp.splitext(f)[1] for f in files)

            bundler_name = bundler_ver.split("@")[0]

            if helpers and "lib_split_label" in helpers:
                split_val = helpers["lib_split_label"](lib_ver)
            else:
                raw = lib_split.get(lib_ver, "unknown")
                split_val = (
                    str(next(iter(raw.values()), "unknown"))
                    if isinstance(raw, dict) else str(raw)
                )
                split_val = "val" if split_val == "valid" else split_val

            rows.append({
                "lib_ver"    : lib_ver,
                "bundler"    : bundler_name,
                "bundler_ver": bundler_ver,
                "split"      : split_val,
                "n_graphs"   : n_graphs,
                "file_types" : ", ".join(sorted(exts)) or "—",
            })

    n_libs     = len(set(r["lib_ver"] for r in rows))
    n_bundlers = len(set(r["bundler"] for r in rows))
    n_combos   = len(rows)
    n_total    = sum(r["n_graphs"] for r in rows)

    # In entire mode each bundle has at most 1 _program file
    file_noun = "_program file" if load_type == LOAD_TYPE_ENTIRE else "function graph file"

    print(f"\n  Load type         : {load_type}")
    print(f"  Lib versions      : {n_libs}")
    print(f"  Unique bundlers   : {n_bundlers}  "
          f"({', '.join(sorted(set(r['bundler'] for r in rows)))})")
    print(f"  lib × bundler     : {n_combos}")
    print(f"  Total {file_noun}s: {n_total}")

    if load_type == LOAD_TYPE_ENTIRE:
        missing = [r for r in rows if r["n_graphs"] == 0]
        if missing:
            print(f"\n  ⚠  {len(missing)} bundles have NO _program file:")
            for r in missing[:10]:
                print(f"     {r['lib_ver']}/{r['bundler_ver']}")
            if len(missing) > 10:
                print(f"     ... and {len(missing)-10} more")

    print(f"\n  {'lib@ver':<30} {'bundler':<12} {'split':<8} {'files':>6}  types")
    print(f"  {SEP}")
    for r in rows:
        print(f"  {r['lib_ver']:<30} {r['bundler']:<12} {r['split']:<8} "
              f"{r['n_graphs']:>6}  {r['file_types']}")

    return rows


# =============================================================================
# Section 2 — Split distribution
# =============================================================================

def section_split(rows: list, save_dir: str, load_type: str = LOAD_TYPE_INDIVIDUAL):
    print(f"\n{SEP2}")
    print(f"  SECTION 2 — SPLIT DISTRIBUTION  [{load_type}]")
    print(SEP2)

    split_libs   = defaultdict(set)
    split_graphs = defaultdict(int)
    for r in rows:
        split_libs[r["split"]].add(r["lib_ver"])
        split_graphs[r["split"]] += r["n_graphs"]

    file_noun = "_program files" if load_type == LOAD_TYPE_ENTIRE else "graphs"

    print(f"\n  {'split':<10} {'libs':>6} {file_noun:>12}")
    print(f"  {SEP[:34]}")
    for sp in ("train", "val", "test", "unknown"):
        if sp in split_libs:
            print(f"  {sp:<10} {len(split_libs[sp]):>6} "
                  f"{split_graphs[sp]:>12}")
    total_g = sum(split_graphs.values())
    total_l = sum(len(v) for v in split_libs.values())
    print(f"  {'TOTAL':<10} {total_l:>6} {total_g:>12}")

    if not HAS_MPL:
        return

    splits    = [s for s in ("train","val","test","unknown") if s in split_libs]
    lib_vals  = [len(split_libs[s])  for s in splits]
    grph_vals = [split_graphs[s]     for s in splits]
    colors    = ["#4C72B0","#DD8452","#55A868","#aaa"][:len(splits)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title in zip(
        axes,
        [lib_vals, grph_vals],
        ["Lib versions per split", f"{file_noun.capitalize()} per split"],
    ):
        wedges, texts, autotexts = ax.pie(
            vals, labels=splits, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(title, fontsize=11, pad=12)

    fig.suptitle(f"Train / Val / Test Split  [{load_type}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "02_split_distribution.png")


# =============================================================================
# Section 3 — Graph sizes
# =============================================================================

def section_sizes(
    raw_dir: str,
    rows: list,
    save_dir: str,
    sample_per_bundler: int = 40,
    helpers: dict = None,
    data_dir: str = None,
    lib_filter: list = None,
    load_type: str = LOAD_TYPE_INDIVIDUAL,
):
    print(f"\n{SEP2}")
    print(f"  SECTION 3 — GRAPH SIZE DISTRIBUTION  [{load_type}]")
    print(SEP2)
    cap_msg = (
        f"  (loading the single _program file per bundle)"
        if load_type == LOAD_TYPE_ENTIRE
        else f"  (sampling up to {sample_per_bundler} graphs per lib×bundler combo)"
    )
    print(cap_msg)

    size_rows  = []
    lib_filter = lib_filter or []

    for r in rows:
        if not _lib_matches(r["lib_ver"], lib_filter):
            continue
        graph_root = data_dir or raw_dir
        graphs_dir = osp.join(graph_root, r["lib_ver"], r["bundler_ver"], "graphs")
        if not osp.isdir(graphs_dir):
            continue

        fnames = get_graph_files(graphs_dir, load_type)
        if load_type == LOAD_TYPE_INDIVIDUAL:
            fnames = fnames[:sample_per_bundler]

        for fname in fnames:
            G = load_graph(osp.join(graphs_dir, fname))
            if G is None:
                continue

            if helpers and "graph_split_label" in helpers:
                g_split = helpers["graph_split_label"](
                    r["lib_ver"], r["bundler_ver"], fname)
            else:
                g_split = r["split"]

            joern = _is_joern_graph(G)
            # For entire mode, also report function-entry count
            if load_type == LOAD_TYPE_ENTIRE and joern:
                n_methods = sum(
                    1 for n in G.nodes()
                    if (_safe_int(n) >> 32) == 1
                )
            elif load_type == LOAD_TYPE_ENTIRE:
                n_methods = sum(
                    1 for _, attr in G.nodes(data=True)
                    if any(kw.upper() in attr.get("label","").upper()
                           for kw in ("METHOD","FUNCTION","FunctionDeclaration"))
                )
            else:
                n_methods = None

            size_rows.append({
                "lib_ver" : r["lib_ver"],
                "bundler" : r["bundler"],
                "split"   : g_split,
                "fname"   : fname,
                "nodes"   : G.number_of_nodes(),
                "edges"   : G.number_of_edges(),
                "joern"   : joern,
                "n_methods": n_methods,
            })

    if not size_rows:
        print("  [WARN] No graphs could be loaded for size analysis.")
        return size_rows

    nodes = np.array([r["nodes"]    for r in size_rows])
    edges = np.array([r["edges"]    for r in size_rows])
    joern_count = sum(1 for r in size_rows if r["joern"])

    print(f"\n  Sampled {len(size_rows)} file(s)  |  "
          f"Joern format: {joern_count}/{len(size_rows)}\n")

    for name, arr in [("nodes", nodes), ("edges", edges)]:
        print(f"  {name}:")
        print(f"    min={arr.min()}  p25={int(np.percentile(arr,25))}  "
              f"median={int(np.median(arr))}  p75={int(np.percentile(arr,75))}  "
              f"p99={int(np.percentile(arr,99))}  max={arr.max()}")

    # In entire mode, also print function-entry count stats
    if load_type == LOAD_TYPE_ENTIRE:
        methods = np.array([r["n_methods"] for r in size_rows if r["n_methods"] is not None])
        if len(methods):
            print(f"\n  function-entry nodes per _program file:")
            print(f"    min={methods.min()}  median={int(np.median(methods))}  "
                  f"max={methods.max()}  total={methods.sum()}")
            print(f"    (→ subgraphs extracted at max_depth=N will be ≤ this count)")

    if not HAS_MPL:
        return size_rows

    # ── histogram ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, arr, col, color in zip(
        axes, [nodes, edges], ["nodes", "edges"], ["#4C72B0", "#DD8452"],
    ):
        clip = int(np.percentile(arr, 99))
        vals = np.clip(arr, 0, clip)
        ax.hist(vals, bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(arr), color="black", linestyle="--",
                   linewidth=1.2, label=f"median = {int(np.median(arr))}")
        ax.axvline(np.mean(arr),   color="red",   linestyle=":",
                   linewidth=1.0, label=f"mean   = {int(np.mean(arr))}")
        ax.set_xlabel(col.capitalize(), fontsize=10)
        ax.set_ylabel("# files", fontsize=10)
        ax.set_title(
            f"{col.capitalize()} per {'_program file' if load_type == LOAD_TYPE_ENTIRE else 'graph'}"
            f"  (clipped at p99={clip})",
            fontsize=10,
        )
        ax.legend(fontsize=8)

    fig.suptitle(f"Graph Size Distribution  [{load_type}]",
                 fontsize=13, fontweight="bold")
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
    ax.set_title(
        f"Node count per lib  [{load_type}]  "
        "(box = IQR, whiskers = 1.5×IQR)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "03b_nodes_per_lib.png")

    # ── extra: function-entry count per lib (entire mode only) ─────────────────
    if load_type == LOAD_TYPE_ENTIRE:
        lib_methods = [
            [r["n_methods"] for r in size_rows
             if r["lib_ver"] == lib and r["n_methods"] is not None]
            for lib in libs_uniq
        ]
        if any(lib_methods):
            fig, ax = plt.subplots(figsize=(max(8, len(libs_uniq) * 0.9), 5))
            bp = ax.boxplot(
                [m if m else [0] for m in lib_methods],
                patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 1.5},
            )
            for patch, color in zip(bp["boxes"],
                                     plt.cm.tab20.colors[:len(libs_uniq)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(libs_uniq) + 1))
            ax.set_xticklabels(libs_uniq, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("METHOD / FUNCTION nodes", fontsize=10)
            ax.set_title(
                "Function-entry nodes per _program file  (= max subgraphs extractable)",
                fontsize=11,
            )
            plt.tight_layout()
            _save_or_show(fig, save_dir, "03c_methods_per_lib.png")

    return size_rows


# =============================================================================
# Section 4 — Edge types
# =============================================================================

def section_edges(
    raw_dir: str,
    rows: list,
    save_dir: str,
    max_graphs: int = 200,
    data_dir: str = None,
    lib_filter: list = None,
    load_type: str = LOAD_TYPE_INDIVIDUAL,
):
    print(f"\n{SEP2}")
    print(f"  SECTION 4 — EDGE TYPE DISTRIBUTION  [{load_type}]")
    print(SEP2)
    cap_msg = (
        "  (loading _program file per bundle — may be slow for large CPGs)"
        if load_type == LOAD_TYPE_ENTIRE
        else f"  (sampling up to {max_graphs} graphs total)"
    )
    print(cap_msg)

    edge_counter  = Counter()
    group_counter = Counter()
    sampled       = 0
    lib_filter    = lib_filter or []

    for r in rows:
        if load_type == LOAD_TYPE_INDIVIDUAL and sampled >= max_graphs:
            break
        if not _lib_matches(r["lib_ver"], lib_filter):
            continue
        graph_root = data_dir or raw_dir
        graphs_dir = osp.join(graph_root, r["lib_ver"], r["bundler_ver"], "graphs")
        if not osp.isdir(graphs_dir):
            continue

        fnames = get_graph_files(graphs_dir, load_type)
        # For individual mode, cap per bundle at 10 to stay under max_graphs
        if load_type == LOAD_TYPE_INDIVIDUAL:
            fnames = fnames[:10]

        for fname in fnames:
            if load_type == LOAD_TYPE_INDIVIDUAL and sampled >= max_graphs:
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
    print(f"\n  Sampled {sampled} file(s)  |  {total:,} edges total\n")

    if not total:
        print("  [WARN] No edges found — check load_type and graph files.")
        return

    print(f"  {'edge_type':<22} {'count':>8}  {'%':>6}")
    print(f"  {SEP[:42]}")
    for etype, cnt in edge_counter.most_common():
        print(f"  {etype:<22} {cnt:>8,}  {cnt/total*100:>5.1f}%")

    print(f"\n  {'group':<22} {'count':>8}  {'%':>6}")
    print(f"  {SEP[:42]}")
    for grp, cnt in group_counter.most_common():
        print(f"  {grp:<22} {cnt:>8,}  {cnt/total*100:>5.1f}%")

    if not HAS_MPL or not edge_counter:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_n   = 15
    items   = edge_counter.most_common(top_n)
    e_names = [i[0] for i in items]
    e_vals  = [i[1] for i in items]
    colors  = [EDGE_COLORS.get(EDGE_GROUPS.get(n, 0), "#aaa") for n in e_names]
    axes[0].barh(e_names[::-1], e_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].set_title(f"Top {top_n} edge types", fontsize=11)
    axes[0].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    grp_items  = sorted(group_counter.items(), key=lambda x: -x[1])
    g_names    = [i[0] for i in grp_items]
    g_vals     = [i[1] for i in grp_items]
    grp_colors = [
        EDGE_COLORS.get(
            next((k for k, v in GROUP_NAMES.items() if v == n), 0), "#aaa"
        )
        for n in g_names
    ]
    axes[1].barh(g_names[::-1], g_vals[::-1], color=grp_colors[::-1])
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].set_title("Edge group distribution", fontsize=11)
    axes[1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    fig.suptitle(f"Edge Type Analysis  [{load_type}]",
                 fontsize=13, fontweight="bold")
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
        print(f"  Run:  python -m graphgps.loader.dataset.cpg_vocab "
              f"--raw_dir {raw_dir}")
        return

    with open(vocab_path) as f:
        vocab = json.load(f)

    print(f"\n  Vocab size: {len(vocab)} entries")

    if len(vocab) <= 1:
        print("\n  [WARN] Only UNK in vocab — label extraction likely failed.")
        print(f"  Run:  python -m graphgps.loader.dataset.cpg_vocab "
              f"--raw_dir {raw_dir} --inspect")
        return

    items = list(vocab.items())[1:]   # skip UNK=0
    print(f"\n  {'rank':<6} {'label':<35} {'idx':>5}")
    print(f"  {SEP[:50]}")
    for rank, (label, idx) in enumerate(items[:40], start=1):
        print(f"  {rank:<6} {label:<35} {idx:>5}")
    if len(items) > 40:
        print(f"  ... ({len(items) - 40} more entries)")

    if not HAS_MPL or len(items) < 2:
        return

    top         = items[:30]
    labels_plot = [x[0][:22] for x in top]
    freq_rank   = list(range(len(top), 0, -1))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.barh(
        labels_plot[::-1], freq_rank[::-1],
        color=plt.cm.viridis(np.linspace(0.2, 0.85, len(top))),
    )
    ax.set_xlabel("Relative frequency rank  (higher = more frequent)", fontsize=10)
    ax.set_title(
        "Top 30 CPG node label types  (sorted by corpus frequency)", fontsize=11
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "05_vocab_top30.png")


# =============================================================================
# Section 6 — Sample graph visualisation
# =============================================================================

def section_graphs(
    raw_dir: str,
    rows: list,
    save_dir: str,
    n_samples: int = 3,
    max_nodes: int = 80,
    helpers: dict = None,
    data_dir: str = None,
    lib_filter: list = None,
    load_type: str = LOAD_TYPE_INDIVIDUAL,
):
    print(f"\n{SEP2}")
    print(f"  SECTION 6 — SAMPLE GRAPH VISUALISATIONS  [{load_type}]")
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

        for fname in get_graph_files(graphs_dir, load_type):
            G = load_graph(osp.join(graphs_dir, fname))
            if G is None or G.number_of_nodes() < 4:
                continue

            if helpers and "graph_split_label" in helpers:
                g_split = helpers["graph_split_label"](
                    r["lib_ver"], r["bundler_ver"], fname)
            else:
                g_split = r["split"]

            title = (
                f"{r['lib_ver']}  /  {r['bundler_ver']}  /  {fname}"
                f"  [{g_split}]  [{load_type}]"
            )
            _visualise_graph(
                G, title, max_nodes, save_dir,
                fname=f"06_graph_{shown+1:02d}.png",
                load_type=load_type,
            )
            shown += 1
            break

    if shown == 0:
        print("  [WARN] No graphs found to visualise.")


def _visualise_graph(
    G, title: str, max_nodes: int, save_dir: str,
    fname: str, load_type: str = LOAD_TYPE_INDIVIDUAL,
):
    joern = _is_joern_graph(G)
    info  = (
        f"  {G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges"
        f"{'  [Joern format]' if joern else ''}"
    )
    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G     = G.subgraph(nodes)
        info += f"  (truncated to {max_nodes} nodes)"
    print(f"\n  {title}")
    print(info)

    pos = nx.spring_layout(G, seed=42,
                            k=1.8 / max(1, G.number_of_nodes() ** 0.5))

    edge_groups = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        etype = attr.get("label") or attr.get("type") or "AST"
        edge_groups[EDGE_GROUPS.get(etype, 0)].append((u, v))

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f8f8f8")

    # Colour nodes differently for entire mode: highlight METHOD nodes
    if load_type == LOAD_TYPE_ENTIRE:
        node_colors = []
        for n, attr in G.nodes(data=True):
            lbl = _get_node_label(n, attr)
            if lbl == "METHOD":
                node_colors.append("#E24B4A")   # red = function entry
            elif lbl in ("CALL", "METHOD_REF"):
                node_colors.append("#f0883e")   # amber = call site
            else:
                node_colors.append("#AEC6CF")   # default
    else:
        node_colors = "#AEC6CF"

    nx.draw_networkx_nodes(
        G, pos, node_size=90, node_color=node_colors, alpha=0.92, ax=ax,
    )
    for grp, edges in edge_groups.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            edge_color=EDGE_COLORS.get(grp, "#999"),
            arrows=True, arrowsize=10,
            width=0.9, alpha=0.65, ax=ax,
        )

    # Node labels: use semantic type for Joern, text label for others
    node_labels = {}
    for n, attr in G.nodes(data=True):
        lbl = _get_node_label(n, attr)
        node_labels[n] = lbl[:16] + "…" if len(lbl) > 16 else lbl
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5, ax=ax)

    legend_handles = [
        Patch(color=EDGE_COLORS[g], label=GROUP_NAMES[g])
        for g in sorted(edge_groups.keys())
    ]
    if load_type == LOAD_TYPE_ENTIRE:
        legend_handles += [
            Patch(color="#E24B4A", label="METHOD (anchor)"),
            Patch(color="#f0883e", label="CALL / METHOD_REF"),
            Patch(color="#AEC6CF", label="other nodes"),
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
                        help="raw/ directory — split.json and vocab always read here")
    parser.add_argument("--data_dir",  default=None,
                        help="Directory containing lib@ver/ graph subdirs. "
                             "Defaults to --raw_dir when not set.")
    parser.add_argument("--load_type", default=LOAD_TYPE_INDIVIDUAL,
                        choices=LOAD_TYPES,
                        help=(
                            "individual: per-function graph files (excludes _program.*). "
                            "entire: whole-program CPG only (_program.xml / _program.dot)."
                        ))
    parser.add_argument("--bundler",   nargs="+", default=[], metavar="BUNDLER",
                        help="Filter to specific bundler versions, e.g. "
                             "rollup@4.46.2 webpack@5.95.0. Default: all.")
    parser.add_argument("--lib",       nargs="+", default=[], metavar="LIB",
                        help="Filter to specific lib versions, e.g. "
                             "axios lodash chalk@5.3.0. Default: all.")
    parser.add_argument("--sections",  nargs="+", default=ALL_SECTIONS,
                        choices=ALL_SECTIONS, metavar="SECTION",
                        help=f"Sections to run. Default: all. "
                             f"Choices: {ALL_SECTIONS}")
    parser.add_argument("--save_dir",  default="",
                        help="Directory to save figures (empty = show interactively)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of sample graphs to visualise")
    parser.add_argument("--max_nodes", type=int, default=80,
                        help="Max nodes to render per sample graph")
    parser.add_argument("--sample_per_bundler", type=int, default=40,
                        help="(individual mode) Max graphs per lib×bundler for sizes")
    parser.add_argument("--edge_sample", type=int, default=200,
                        help="(individual mode) Max graphs for edge type analysis")
    args = parser.parse_args()

    if not osp.isdir(args.raw_dir):
        sys.exit(f"[ERROR] raw_dir not found: {args.raw_dir}")
    if args.data_dir and not osp.isdir(args.data_dir):
        sys.exit(f"[ERROR] data_dir not found: {args.data_dir}")

    data_dir       = args.data_dir or None
    bundler_filter = args.bundler  or []
    lib_filter     = args.lib      or []

    split_json = osp.join(args.raw_dir, "split.json")
    if not osp.exists(split_json):
        sys.exit(f"[ERROR] split.json not found: {split_json}\n"
                 "  Run build_split.py first.")
    with open(split_json) as f:
        lib_split = json.load(f)

    first_val  = next(iter(lib_split.values()), None)
    split_mode = "closed" if isinstance(first_val, dict) else "open"

    def _lib_split_label(lib_ver: str) -> str:
        val = lib_split.get(lib_ver, "unknown")
        if isinstance(val, dict):
            counts = Counter(v for v in val.values() if isinstance(v, str))
            sp     = counts.most_common(1)[0][0] if counts else "unknown"
        else:
            sp = str(val)
        return "val" if sp == "valid" else sp

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
            matplotlib.use("Agg")

    print(f"\n{SEP2}")
    print("  JSLibs EDA")
    print(f"  raw_dir   : {osp.abspath(args.raw_dir)}")
    print(f"  graph_root: {osp.abspath(data_dir or args.raw_dir)}")
    print(f"  load_type : {args.load_type}")
    print(f"  split mode: {split_mode}")
    print(f"  bundlers  : {bundler_filter or 'ALL'}")
    print(f"  libs      : {lib_filter or 'ALL'}")
    print(f"  sections  : {args.sections}")
    print(SEP2)

    helpers = {
        "lib_split_label"  : _lib_split_label,
        "graph_split_label": _graph_split_label,
        "split_mode"       : split_mode,
    }

    rows = []
    if "overview" in args.sections:
        rows = section_overview(
            args.raw_dir, lib_split, args.save_dir,
            helpers=helpers, data_dir=data_dir,
            bundler_filter=bundler_filter, lib_filter=lib_filter,
            load_type=args.load_type,
        )

    if not rows:
        rows = section_overview(
            args.raw_dir, lib_split, save_dir="",
            helpers=helpers, data_dir=data_dir,
            bundler_filter=bundler_filter, lib_filter=lib_filter,
            load_type=args.load_type,
        )

    if "split" in args.sections:
        section_split(rows, args.save_dir, load_type=args.load_type)

    if "sizes" in args.sections:
        section_sizes(
            args.raw_dir, rows, args.save_dir,
            sample_per_bundler=args.sample_per_bundler,
            helpers=helpers, data_dir=data_dir, lib_filter=lib_filter,
            load_type=args.load_type,
        )

    if "edges" in args.sections:
        section_edges(
            args.raw_dir, rows, args.save_dir,
            max_graphs=args.edge_sample, data_dir=data_dir,
            lib_filter=lib_filter, load_type=args.load_type,
        )

    if "vocab" in args.sections:
        section_vocab(args.raw_dir, args.save_dir)

    if "graphs" in args.sections:
        section_graphs(
            args.raw_dir, rows, args.save_dir,
            n_samples=args.n_samples, max_nodes=args.max_nodes,
            helpers=helpers, data_dir=data_dir, lib_filter=lib_filter,
            load_type=args.load_type,
        )

    print(f"\n{SEP2}")
    print("  EDA complete.")
    if args.save_dir:
        print(f"  Figures saved to: {osp.abspath(args.save_dir)}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()