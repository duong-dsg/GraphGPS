"""
graphgps/loader/dataset/jslibs.py

Loading strategy
----------------
Instead of reading one file per function, this version reads the single
whole-program CPG (_program.xml or _program.dot) produced by the CPG
extractor, locates every FUNCTION / METHOD entry node inside it, then
extracts a k-hop ego-subgraph around each such node.

Each extracted subgraph becomes one torch_geometric.data.Data object and
is labelled with the parent library class index — exactly the same label
space and split contract as before.

Why this is better
------------------
* The whole-program CPG contains inter-function edges (CALL, CDG, REF, ...)
  that are lost when each function is stored in isolation.  The k-hop window
  captures local inter-function context without blowing up graph size.
* No per-function file naming conventions to maintain.
* max_depth controls the receptive field analogously to GNN depth, so the
  hyperparameter has an interpretable meaning.

Split schemas (unchanged from v1)
-----------------------------------
closed  {"axios@1.7.9": {"rollup@4.46.2/graphs/_program.xml": "train", ...}}
open    {"axios@1.7.9": "train", ...}

The split key lookup tries both ``_program.xml`` and ``_program.dot`` so
either extension works.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import pydot
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data, InMemoryDataset

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

EDGE_GROUPS: Dict[str, int] = {
    "AST": 0,        "CONTAINS": 0,
    "CFG": 1,        "DOMINATE": 1,   "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4,       "ARGUMENT": 4,   "PARAMETER_LINK": 4,
    "REF": 5,
}
NUM_EDGE_GROUPS  = len(set(EDGE_GROUPS.values()))   # 6
NODE_FEATURE_DIM = 128

# Node label substrings that identify function-entry nodes in a CPG.
# Joern/codepropertygraph uses "METHOD"; some tools use "FUNCTION".
# The check is case-insensitive substring match so "FunctionDeclaration"
# and "method_definition" both match.
FUNCTION_NODE_KEYWORDS: Tuple[str, ...] = (
    "METHOD",
    "FUNCTION",
    "FunctionDeclaration",
    "ArrowFunctionExpression",
    "FunctionExpression",
)

# File names (without directory) that carry the whole-program CPG.
PROGRAM_FILE_STEMS: Tuple[str, ...] = ("_program",)


# =============================================================================
# Split schema detection
# =============================================================================

def detect_split_mode(lib_split: Dict) -> str:
    first = next(iter(lib_split.values()))
    if isinstance(first, dict):
        return "closed"
    if isinstance(first, str):
        return "open"
    raise ValueError(
        f"Unrecognised split.json schema — expected str or dict, got {type(first)}"
    )


# =============================================================================
# Graph I/O  (reads _program.* files only)
# =============================================================================

def _read_dot(path: str) -> nx.MultiDiGraph:
    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise ValueError(f"pydot returned empty list: {path}")
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


def _read_xml(path: str) -> nx.MultiDiGraph:
    for reader in (nx.read_graphml, nx.read_gexf):
        try:
            return nx.MultiDiGraph(reader(path))
        except Exception:
            pass
    # manual fallback
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
        if src is None or dst is None:
            continue
        attrs = dict(edge.attrib)
        for d in edge.findall(".//data"):
            if d.get("key") and d.text:
                attrs[d.get("key")] = d.text
        G.add_edge(src, dst, **attrs)
    return G


def load_program_graph(graphs_dir: str) -> Tuple[Optional[nx.MultiDiGraph], Optional[str]]:
    """
    Look for a _program.xml or _program.dot file in graphs_dir.
    Returns (graph, filename) or (None, None) if not found.
    """
    for stem in PROGRAM_FILE_STEMS:
        for ext in (".xml", ".dot"):
            fname = stem + ext
            fpath = osp.join(graphs_dir, fname)
            if osp.isfile(fpath):
                try:
                    if ext == ".dot":
                        return _read_dot(fpath), fname
                    else:
                        return _read_xml(fpath), fname
                except Exception as exc:
                    log.warning("Failed to parse %s: %s", fpath, exc)
                    return None, fname   # file exists but broken
    return None, None


# =============================================================================
# K-hop ego subgraph extraction
# =============================================================================

def _is_function_node(label: str) -> bool:
    """Return True if node label indicates a function/method entry."""
    label_up = label.upper()
    return any(kw.upper() in label_up for kw in FUNCTION_NODE_KEYWORDS)


def _bfs_neighborhood(
    G: nx.MultiDiGraph,
    root: str,
    max_depth: int,
) -> Set[str]:
    """
    BFS on the *undirected* view of G (both in- and out-edges) up to max_depth
    hops from root.  Returns the set of node ids in the neighborhood
    (including root itself).

    Using the undirected view ensures we capture upstream AST parents and
    downstream CFG children with a single BFS, mirroring what a GNN with
    bidirectional message-passing sees.
    """
    visited: Set[str] = {root}
    queue: deque[Tuple[str, int]] = deque([(root, 0)])
    # nx.MultiDiGraph.to_undirected() is expensive on large graphs;
    # we iterate successors + predecessors manually instead.
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return visited


def extract_function_subgraphs(
    G: nx.MultiDiGraph,
    max_depth: int = 3,
    min_nodes: int = 5,
    max_nodes: int = 2000,
) -> List[Tuple[str, nx.MultiDiGraph]]:
    """
    For each function-entry node in G, extract its k-hop ego subgraph.

    Parameters
    ----------
    G          : whole-program CPG
    max_depth  : BFS hop limit (analogous to number of GNN layers)
    min_nodes  : drop subgraphs smaller than this
    max_nodes  : drop subgraphs larger than this (degenerate whole-program nodes)

    Returns
    -------
    List of (anchor_node_id, subgraph) pairs, one per function entry node
    that passes the size filter.
    """
    results: List[Tuple[str, nx.MultiDiGraph]] = []

    # Collect function-entry nodes
    entry_nodes = [
        node for node, attr in G.nodes(data=True)
        if _is_function_node(attr.get("label", ""))
    ]

    if not entry_nodes:
        # Fallback: if CPG has no recognisable function nodes, treat the
        # entire graph as one subgraph so we never silently drop a bundle.
        log.debug("No function-entry nodes found — using whole graph as single subgraph")
        if min_nodes <= G.number_of_nodes() <= max_nodes:
            results.append(("__whole__", G))
        return results

    for anchor in entry_nodes:
        neighborhood = _bfs_neighborhood(G, anchor, max_depth)
        n = len(neighborhood)
        if n < min_nodes or n > max_nodes:
            continue
        sub = G.subgraph(neighborhood).copy()
        results.append((anchor, sub))

    return results


# =============================================================================
# Feature encoding  (unchanged from v1)
# =============================================================================

def _encode_nodes(
    G: nx.MultiDiGraph,
    node2id: Dict,
    vocab: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    vocab provided  → [N, 1] long tensor  (for nn.Embedding)
    vocab=None      → [N, NODE_FEATURE_DIM] float one-hot hash  (legacy)
    """
    if vocab is not None:
        indices = [
            vocab.get(attr.get("label", "UNK"), 0)
            for _, attr in G.nodes(data=True)
        ]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(1)

    x = torch.zeros((len(node2id), NODE_FEATURE_DIM))
    for node, attr in G.nodes(data=True):
        h = hash(attr.get("label", "UNK")) % NODE_FEATURE_DIM
        x[node2id[node]][h] = 1.0
    return x


def _encode_edges(
    G: nx.MultiDiGraph,
    node2id: Dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    ei: List[List[int]] = []
    ea: List[List[int]] = []
    for u, v, attr in G.edges(data=True):
        if u not in node2id or v not in node2id:
            continue
        src = node2id[u];  dst = node2id[v]
        grp = EDGE_GROUPS.get(
            attr.get("label") or attr.get("type") or "AST", 0
        )
        ei += [[src, dst], [dst, src]]
        ea += [[grp, 0],   [grp, 1]]
    if not ei:
        return None, None
    return (
        torch.tensor(ei, dtype=torch.long).t().contiguous(),
        torch.tensor(ea, dtype=torch.long),
    )


def nx_to_pyg(
    G: nx.MultiDiGraph,
    label: int,
    vocab: Optional[Dict[str, int]] = None,
) -> Optional[Data]:
    if G.number_of_nodes() == 0:
        return None
    node2id               = {n: i for i, n in enumerate(G.nodes())}
    x                     = _encode_nodes(G, node2id, vocab=vocab)
    edge_index, edge_attr = _encode_edges(G, node2id)
    if edge_index is None:
        return None
    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        num_nodes  = len(node2id),
        y          = torch.tensor([label], dtype=torch.long),
    )


# =============================================================================
# Helpers
# =============================================================================

def _split_bundler_ver(bundler_ver: str) -> Tuple[str, str]:
    if "@" in bundler_ver:
        name, ver = bundler_ver.split("@", 1)
        return name, ver
    return bundler_ver, ""


def _resolve_program_split_key(
    mode: str,
    lib_info,           # str (open) or dict (closed)
    bundler_ver: str,
    fname: str,
) -> Optional[str]:
    """
    Open mode  : return the lib-level split string directly.
    Closed mode: look up the _program file key in lib_info and return its
                 split string.  Tries both .xml and .dot extensions.
                 Returns None if the file is not listed in split.json.
    """
    if mode == "open":
        key = lib_info if isinstance(lib_info, str) else None
    else:
        stem = osp.splitext(fname)[0]
        candidates = [
            f"{bundler_ver}/graphs/{fname}",
            f"{bundler_ver}/graphs/{stem}.xml",
            f"{bundler_ver}/graphs/{stem}.dot",
        ]
        key = None
        for c in candidates:
            if c in lib_info:
                key = lib_info[c]
                break

    if key is None:
        return None
    return "val" if key in ("valid", "val") else key


def _assign_subgraph_splits(
    n_subgraphs: int,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float   = 0.15,
) -> List[str]:
    """
    Closed mode — post-extraction split assignment.

    The _program file is a single atomic unit in split.json (one key → one
    split label), so all its subgraphs would naively receive the same label,
    making val/test sets empty.  Instead we randomly partition the subgraphs
    extracted from *each* program file into train / val / test according to
    configurable ratios.

    Deterministic: seeded on (lib_idx * 1000 + bundler_hash) so the same
    data always produces the same partition across re-runs.

    Parameters
    ----------
    n_subgraphs  : number of subgraphs extracted from this program file
    seed         : per-bundle integer seed for reproducibility
    train_ratio  : fraction assigned to train  (default 0.70)
    val_ratio    : fraction assigned to val    (default 0.15)
                   remainder → test

    Returns
    -------
    List[str] of length n_subgraphs, each element "train" | "val" | "test"
    """
    import random
    rng = random.Random(seed)
    indices = list(range(n_subgraphs))
    rng.shuffle(indices)

    n_train = max(1, int(n_subgraphs * train_ratio))
    n_val   = max(1, int(n_subgraphs * val_ratio))
    # ensure we never exceed n_subgraphs
    if n_train + n_val >= n_subgraphs and n_subgraphs >= 3:
        n_val = 1
    if n_train + n_val >= n_subgraphs:
        n_train = n_subgraphs - 1
        n_val   = 0

    split_keys = [""] * n_subgraphs
    for rank, orig_idx in enumerate(indices):
        if rank < n_train:
            split_keys[orig_idx] = "train"
        elif rank < n_train + n_val:
            split_keys[orig_idx] = "val"
        else:
            split_keys[orig_idx] = "test"

    return split_keys


# =============================================================================
# Dataset
# =============================================================================

class JSLibsDataset(InMemoryDataset):
    """
    Graph-classification dataset for JS library fingerprinting.

    Each sample is a k-hop ego subgraph extracted from the whole-program CPG
    (_program.xml / _program.dot), centred on a function-entry node.

    Parameters
    ----------
    root                  : dataset root (processed/ lives here)
    data_dir              : directory containing lib@ver/ subdirectories.
                            Defaults to <root>/raw/.
    split_path            : path to split.json
    max_depth             : BFS hop radius for ego subgraph extraction
    min_nodes             : minimum nodes per subgraph (smaller → dropped)
    max_nodes             : maximum nodes per subgraph (larger → dropped)
    max_graphs_per_bundler: cap the number of function subgraphs kept per
                            bundler (useful for class-balance during debug)
    """

    def __init__(
        self,
        root: str,
        data_dir: Optional[str]               = None,
        split_path: Optional[str]             = None,
        max_depth: int                        = 3,
        min_nodes: int                        = 5,
        max_nodes: int                        = 2000,
        max_graphs_per_bundler: Optional[int] = None,
        closed_train_ratio: float             = 0.70,
        closed_val_ratio: float               = 0.15,
        transform: Optional[Callable]         = None,
        pre_transform: Optional[Callable]     = None,
        pre_filter: Optional[Callable]        = None,
    ):
        self._data_dir              = data_dir
        self.split_path             = split_path or osp.join(root, "raw", "split.json")
        self.max_depth              = max_depth
        self.min_nodes              = min_nodes
        self.max_nodes              = max_nodes
        self.max_graphs_per_bundler = max_graphs_per_bundler
        self.closed_train_ratio     = closed_train_ratio
        self.closed_val_ratio       = closed_val_ratio
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_dir(self):       return osp.join(self.root, "raw")
    @property
    def processed_dir(self): return osp.join(self.root, "processed")
    @property
    def graph_dir(self) -> str:
        return self._data_dir if self._data_dir else self.raw_dir
    @property
    def raw_file_names(self): return ["split.json"]
    @property
    def processed_file_names(self): return ["data.pt", "split_dict.pt"]
    @property
    def num_classes(self) -> int:
        return 0 if self.data.y is None else int(self.data.y.max().item()) + 1

    def download(self):
        pass

    # ------------------------------------------------------------------ #
    #  process                                                             #
    # ------------------------------------------------------------------ #

    def process(self):
        # ---- optional vocab ----
        vocab_path = osp.join(self.raw_dir, "cpg_vocab.json")
        vocab: Optional[Dict[str, int]] = None
        if osp.exists(vocab_path):
            with open(vocab_path) as f:
                vocab = json.load(f)
            log.info("Loaded CPG vocab: %d entries", len(vocab))
        else:
            log.warning(
                "cpg_vocab.json not found — using hash node features.\n"
                "Run: python -m graphgps.loader.dataset.cpg_vocab --raw_dir %s",
                self.raw_dir,
            )

        # ---- split.json ----
        with open(self.split_path) as f:
            lib_split: Dict = json.load(f)

        mode = detect_split_mode(lib_split)
        log.info("Split mode: %s", mode)
        print(f"[JSLibs] Split mode: {mode}  |  max_depth={self.max_depth}")

        all_libs   = sorted(lib_split.keys())
        lib_to_idx = {lib: i for i, lib in enumerate(all_libs)}

        VALID_SPLITS = {"train", "val", "test"}

        data_list:  List[Data]      = []
        split_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
        stats = {
            "bundles_seen":    0,
            "bundles_no_prog": 0,   # no _program file found
            "bundles_broken":  0,   # _program exists but parse failed
            "subgraphs_total": 0,
            "subgraphs_loaded":0,
            "skip_size":       0,
            "skip_empty":      0,
            "skip_not_listed": 0,   # closed mode: graph key not in split.json
        }

        graph_root = self.graph_dir
        print(f"[JSLibs] Graph source: {graph_root}")

        for lib_ver in sorted(os.listdir(graph_root)):
            lib_dir = osp.join(graph_root, lib_ver)
            if not osp.isdir(lib_dir) or lib_ver not in lib_split:
                continue

            lib_idx  = lib_to_idx[lib_ver]
            lib_info = lib_split[lib_ver]

            for bundler_ver in sorted(os.listdir(lib_dir)):
                bundler_dir = osp.join(lib_dir, bundler_ver)
                if not osp.isdir(bundler_dir):
                    continue
                graphs_dir = osp.join(bundler_dir, "graphs")
                if not osp.isdir(graphs_dir):
                    continue

                stats["bundles_seen"] += 1
                bundler_name, bundler_version = _split_bundler_ver(bundler_ver)

                # ---- load whole-program CPG ----
                G, prog_fname = load_program_graph(graphs_dir)
                if prog_fname is None:
                    stats["bundles_no_prog"] += 1
                    log.debug("No _program file: %s/%s", lib_ver, bundler_ver)
                    continue
                if G is None:
                    stats["bundles_broken"] += 1
                    log.warning("Parse failed: %s/%s/graphs/%s",
                                lib_ver, bundler_ver, prog_fname)
                    continue

                # ---- open mode: resolve single split key for the whole bundle ----
                # In open mode split.json assigns one split per lib, so every
                # subgraph from this bundle shares the same label.
                if mode == "open":
                    bundle_split_key = _resolve_program_split_key(
                        mode, lib_info, bundler_ver, prog_fname
                    )
                    if bundle_split_key not in VALID_SPLITS:
                        stats["skip_not_listed"] += 1
                        continue

                # ---- extract k-hop subgraphs per function node ----
                subgraphs = extract_function_subgraphs(
                    G,
                    max_depth = self.max_depth,
                    min_nodes = self.min_nodes,
                    max_nodes = self.max_nodes,
                )
                stats["subgraphs_total"] += len(subgraphs)

                if self.max_graphs_per_bundler is not None:
                    subgraphs = subgraphs[: self.max_graphs_per_bundler]

                n = len(subgraphs)
                if n == 0:
                    continue

                # ---- closed mode: assign train/val/test per subgraph ----
                # split.json has one key per _program file, so using that key
                # for all subgraphs would leave val/test empty.  Instead we
                # randomly partition subgraphs from each bundle independently.
                if mode == "closed":
                    bundle_seed = lib_idx * 10007 + hash(bundler_ver) % 9973
                    subgraph_splits = _assign_subgraph_splits(
                        n,
                        seed        = bundle_seed,
                        train_ratio = self.closed_train_ratio,
                        val_ratio   = self.closed_val_ratio,
                    )
                else:
                    # open mode: all subgraphs share the bundle-level key
                    subgraph_splits = [bundle_split_key] * n

                for (anchor_id, sub), split_key in zip(subgraphs, subgraph_splits):
                    if split_key not in VALID_SPLITS:
                        continue

                    data = nx_to_pyg(sub, lib_idx, vocab=vocab)
                    if data is None:
                        stats["skip_empty"] += 1
                        continue

                    # ---- metadata ----
                    data.lib_ver      = lib_ver
                    data.bundler_name = bundler_name
                    data.bundler_ver  = bundler_version
                    data.anchor_node  = str(anchor_id)
                    data.split        = split_key

                    graph_idx = len(data_list)
                    data_list.append(data)
                    split_dict[split_key].append(graph_idx)
                    stats["subgraphs_loaded"] += 1

                log.debug(
                    "%s/%s: %d subgraphs → train=%d val=%d test=%d",
                    lib_ver, bundler_ver, n,
                    subgraph_splits.count("train"),
                    subgraph_splits.count("val"),
                    subgraph_splits.count("test"),
                )

        # ---- summary ----
        print(
            f"[JSLibs] Loaded {stats['subgraphs_loaded']} subgraphs  "
            f"(train={len(split_dict['train'])}  "
            f"val={len(split_dict['val'])}  "
            f"test={len(split_dict['test'])})\n"
            f"         bundles seen={stats['bundles_seen']}  "
            f"no_program={stats['bundles_no_prog']}  "
            f"broken={stats['bundles_broken']}  "
            f"skip_not_listed={stats['skip_not_listed']}\n"
            f"         subgraphs: total_extracted={stats['subgraphs_total']}  "
            f"skip_size={stats['skip_size']}  "
            f"skip_empty={stats['skip_empty']}"
        )

        if not data_list:
            raise RuntimeError(
                "No subgraphs loaded.  Checklist:\n"
                "  1. <graph_dir>/lib@ver/bundler@ver/graphs/_program.xml exists\n"
                "  2. split.json keys match lib@ver directory names exactly\n"
                "  3. _program CPGs contain nodes with FUNCTION/METHOD labels\n"
                f"  4. Split mode detected: '{mode}' — verify split.json schema\n"
                f"  5. max_depth={self.max_depth}  min_nodes={self.min_nodes}  "
                f"max_nodes={self.max_nodes}"
            )

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict,              self.processed_paths[1])

    # ------------------------------------------------------------------ #

    def get_idx_split(self) -> Dict[str, List[int]]:
        d = torch.load(self.processed_paths[1], weights_only=False)
        if "valid" in d and "val" not in d:
            d["val"] = d.pop("valid")
        elif "valid" in d and "val" in d:
            d["val"] += d.pop("valid")
        for k in ("train", "val", "test"):
            d.setdefault(k, [])
        return d

    def __repr__(self) -> str:
        return (
            f"JSLibsDataset(graphs={len(self)}, classes={self.num_classes}, "
            f"max_depth={self.max_depth})"
        )


# =============================================================================
# Label map  (inference helper)
# =============================================================================

def build_label_map(split_json: str) -> Dict[int, str]:
    """idx → lib@ver  (mirrors JSLibsDataset.process() ordering)."""
    with open(split_json) as f:
        lib_split = json.load(f)
    return {i: lib for i, lib in enumerate(sorted(lib_split.keys()))}


# =============================================================================
# Debug entry point
# python -m graphgps.loader.dataset.jslibs
# =============================================================================

if __name__ == "__main__":
    import pprint
    from collections import Counter

    ROOT = "datasets/JSLibs"

    print("=" * 60)
    print("Loading dataset (k-hop subgraph mode)...")
    print("=" * 60)

    dataset = JSLibsDataset(
        root       = ROOT,
        data_dir   = "/home/aiuser4/ado/bundled-js-scan/data/train/v2.2",
        split_path = osp.join(ROOT, "raw", "split.json"),
        max_depth  = 3,
        min_nodes  = 5,
        max_nodes  = 500,
        max_graphs_per_bundler = 20,
    )

    print(dataset)
    print(f"Total subgraphs : {len(dataset)}")
    print(f"Num classes     : {dataset.num_classes}")

    split = dataset.get_idx_split()
    print("\nSplit sizes:")
    pprint.pprint({k: len(v) for k, v in split.items()})

    for name, idxs in split.items():
        if not idxs:
            print(f"⚠️  {name.upper()} split is EMPTY — training will crash")
            continue
        ys = [dataset[i].y.item() for i in idxs]
        print(f"\n{name}: label distribution (top 10)")
        pprint.pprint(Counter(ys).most_common(10))

    print("\n" + "=" * 60)
    print("Sample subgraph [0]")
    print("=" * 60)
    d = dataset[0]
    print(f"lib_ver     : {d.lib_ver}")
    print(f"anchor_node : {d.anchor_node}")
    print(f"split       : {d.split}")
    print(f"num_nodes   : {d.num_nodes}")
    print(f"num_edges   : {d.edge_index.shape[1]}")
    print(f"x shape     : {d.x.shape}")
    print(f"edge_attr   : {d.edge_attr[:6]}")
    print(f"unique edge groups : {set(d.edge_attr[:, 0].tolist())}")

    assert len(split["train"]) > 0, "Train split empty!"
    assert len(split["val"])   > 0, "Val split empty!"
    assert len(split["test"])  > 0, "Test split empty!"
    print("\n✅ All sanity checks passed")