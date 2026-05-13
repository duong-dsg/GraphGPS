"""
graphgps/loader/dataset/jslibs.py

Supports two split.json schemas produced by build_split.py:

  closed mode  (recommended first):
    {
      "axios@1.7.9": {
        "rollup@4.46.2/graphs/func_001.xml": "train",
        "rollup@4.46.2/graphs/func_002.xml": "val",
        ...
      }
    }
    → same lib appears in train + val + test, just different graphs.
    → standard N-class CrossEntropyLoss classifier works correctly.

  open mode:
    { "axios@1.7.9": "train", "lodash@4.17.21": "test", ... }
    → each lib is entirely in one split.
    → requires metric learning at inference; softmax head will give ~0% on test.
"""

import json
import logging
import os
import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import pydot
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from torch_geometric.data import Data, InMemoryDataset

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

EDGE_GROUPS: Dict[str, int] = {
    "AST": 0, "CONTAINS": 0,
    "CFG": 1, "DOMINATE": 1, "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4, "ARGUMENT": 4, "PARAMETER_LINK": 4,
    "REF": 5,
}
NUM_EDGE_GROUPS  = len(set(EDGE_GROUPS.values()))   # 6
NODE_FEATURE_DIM = 128


def _lib_matches(lib_ver: str, lib_filter: List[str], spliter: str = "@") -> bool:
    """True when lib_filter is empty or lib_ver matches an entry.
    Supports exact ('axios@1.7.9') and base-name ('axios') matching."""
    if not lib_filter:
        return True
    base = (spliter + lib_ver.split(spliter)[1]
            if lib_ver.startswith(spliter) else lib_ver.split(spliter)[0])
    return lib_ver in lib_filter or base in lib_filter


# =============================================================================
# Split schema detection
# =============================================================================

def detect_split_mode(lib_split: Dict) -> str:
    """
    Returns "closed" or "open" by inspecting the first value in split.json.
      closed → first value is a dict  {"bundler@ver/graphs/fname": "train", ...}
      open   → first value is a str   "train" | "val" | "test"
    """
    first = next(iter(lib_split.values()))
    if isinstance(first, dict):
        return "closed"
    if isinstance(first, str):
        return "open"
    raise ValueError(
        f"Unrecognised split.json schema — expected str or dict values, got {type(first)}"
    )


# =============================================================================
# Graph I/O
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


def load_graph(path: str) -> nx.MultiDiGraph:
    if path.endswith(".dot"):
        return _read_dot(path)
    if path.endswith(".xml"):
        return _read_xml(path)
    raise ValueError(f"Unsupported extension: {path}")


# =============================================================================
# Feature encoding
# =============================================================================

def _encode_nodes(
    G: nx.MultiDiGraph,
    node2id: Dict,
    vocab: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    vocab provided  → returns [N, 1] long tensor (vocab indices, for nn.Embedding)
    vocab=None      → returns [N, NODE_FEATURE_DIM] float one-hot hash (legacy)
    """
    if vocab is not None:
        indices = [
            vocab.get(attr.get("label", "UNK"), 0)
            for node, attr in G.nodes(data=True)
        ]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(1)
    else:
        x = torch.zeros((len(node2id), NODE_FEATURE_DIM))
        for node, attr in G.nodes(data=True):
            h = hash(attr.get("label", "UNK")) % NODE_FEATURE_DIM
            x[node2id[node]][h] = 1.0
        return x


def _encode_edges(
    G: nx.MultiDiGraph, node2id: Dict
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    ei, ea = [], []
    for u, v, attr in G.edges(data=True):
        if u not in node2id or v not in node2id:
            continue
        src  = node2id[u];  dst = node2id[v]
        grp  = EDGE_GROUPS.get(attr.get("label") or attr.get("type") or "AST", 0)
        ei  += [[src, dst], [dst, src]]
        ea  += [[grp, 0],   [grp, 1]]
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
    """Convert nx.MultiDiGraph → PyG Data with a single integer class label."""
    if G.number_of_nodes() == 0:
        return None
    node2id               = {n: i for i, n in enumerate(G.nodes())}
    x                     = _encode_nodes(G, node2id, vocab=vocab)
    edge_index, edge_attr = _encode_edges(G, node2id)
    if edge_index is None:
        return None
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        num_nodes=len(node2id),
        y=torch.tensor([label], dtype=torch.long),   # [1] long
    )


# =============================================================================
# Bundler dir helpers
# =============================================================================

def _split_bundler_ver(bundler_ver: str) -> Tuple[str, str]:
    if "@" in bundler_ver:
        name, ver = bundler_ver.split("@", 1)
        return name, ver
    return bundler_ver, ""


def _graph_files(graphs_dir: str) -> List[str]:
    return sorted(
        f for f in os.listdir(graphs_dir)
        if (f.endswith((".dot", ".xml"))
            and "Zone.Identifier" not in f
            and not f.startswith("_program"))
    )


# =============================================================================
# Dataset
# =============================================================================

class JSLibsDataset(InMemoryDataset):
    """
    Graph-classification dataset for JS library fingerprinting.

    Reads split.json and auto-detects whether it is closed-set or open-set
    format (see module docstring).  No code change needed when switching modes —
    just regenerate split.json with build_split.py --mode closed|open.
    """

    def __init__(
        self,
        root: str,
        data_dir: Optional[str]            = None,
        split_path: Optional[str]          = None,
        min_nodes: int                     = 5,
        max_nodes: int                     = 2000,
        max_graphs_per_bundler: Optional[int] = None,
        bundler_filter: Optional[List[str]]   = None,
        lib_filter: Optional[List[str]]       = None,
        spliter: Optional[str]             = "@",
        transform: Optional[Callable]      = None,
        pre_transform: Optional[Callable]  = None,
        pre_filter: Optional[Callable]     = None,
    ):
        # data_dir: where lib@ver/ graph directories live.
        # Defaults to raw/ so existing behaviour is unchanged.
        # When set to a custom path, raw/ still holds split.json
        # and cpg_vocab.json; processed/ is under root as usual.
        self._data_dir              = data_dir   # None = use self.raw_dir
        # bundler_filter: optional list of bundler@ver strings to include.
        # e.g. ["rollup@4.46.2", "webpack@5.95.0"]
        # None or [] = include all bundlers (original behaviour).
        self._bundler_filter        = set(bundler_filter) if bundler_filter else None
        # lib_filter: optional list of lib@ver strings to include.
        # Supports exact ('axios@1.7.9') and base-name ('axios') matching.
        # None or [] = include all libs (original behaviour).
        self._lib_filter            = list(lib_filter) if lib_filter else None
        self.spliter                = spliter
        self.split_path             = split_path or osp.join(root, "raw", "split.json")
        self.min_nodes              = min_nodes
        self.max_nodes              = max_nodes
        self.max_graphs_per_bundler = max_graphs_per_bundler
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
        """Root directory containing lib@ver/ graph subdirectories.
        Falls back to raw_dir when data_dir is not set."""
        return self._data_dir if self._data_dir else self.raw_dir
    @property
    def raw_file_names(self): return ["split.json"]
    @property
    def processed_file_names(self): return ["data.pt", "split_dict.pt"]

    @property
    def num_classes(self) -> int:
        if self.data.y is None:
            return 0
        return int(self.data.y.max().item()) + 1

    def download(self):
        pass

    # ------------------------------------------------------------------ #
    #  process                                                             #
    # ------------------------------------------------------------------ #

    def process(self):
        # ---- load vocab if available ----
        vocab_path = osp.join(self.raw_dir, "cpg_vocab.json")
        vocab: Optional[Dict[str, int]] = None
        if osp.exists(vocab_path):
            with open(vocab_path) as f:
                vocab = json.load(f)
            log.info("Loaded CPG vocab: %d entries", len(vocab))
        else:
            log.warning(
                "cpg_vocab.json not found at %s -- using hash node features.\n"
                "Run: python -m graphgps.loader.dataset.cpg_vocab --raw_dir %s",
                vocab_path, self.raw_dir,
            )

        with open(self.split_path) as f:
            lib_split: Dict = json.load(f)

        # normalise "valid" → "val"
        lib_split = {
            k: ({gk: ("val" if gv == "valid" else gv) for gk, gv in v.items()}
                if isinstance(v, dict)
                else ("val" if v == "valid" else v))
            for k, v in lib_split.items()
        }

        mode = detect_split_mode(lib_split)
        log.info("Split mode detected: %s", mode)
        print(f"Split mode: {mode}")

        # label space — sorted lib@ver keys, same order as split.json keys
        all_libs   = sorted(lib_split.keys())
        lib_to_idx = {lib: i for i, lib in enumerate(all_libs)}

        # Canonical split keys — always "train" / "val" / "test"
        VALID_SPLITS = {"train", "val", "test"}

        data_list: List[Data] = []
        split_dict            = {"train": [], "val": [], "test": []}
        stats = {"loaded": 0, "skip_parse": 0, "skip_size": 0,
                 "skip_empty": 0, "skip_not_in_split": 0}

        graph_root = self.graph_dir
        log.info("Graph source directory: %s", graph_root)
        print(f"Graph source: {graph_root}")
        if self._lib_filter:
            print(f"Lib filter    : {sorted(self._lib_filter)}")
        else:
            print("Lib filter    : ALL (no filter)")
        if self._bundler_filter:
            print(f"Bundler filter: {sorted(self._bundler_filter)}")
        else:
            print("Bundler filter: ALL (no filter)")

        # ── diagnostic: show first lib to help debug key mismatches ──────
        dirs_on_disk  = sorted(d for d in os.listdir(graph_root)
                               if osp.isdir(osp.join(graph_root, d)))
        keys_in_split = sorted(lib_split.keys())
        matched       = [d for d in dirs_on_disk if d in lib_split]
        print(f"Libs on disk   : {len(dirs_on_disk)}  "
              f"({dirs_on_disk[:3]}{'...' if len(dirs_on_disk)>3 else ''})")
        print(f"Libs in split  : {len(keys_in_split)}  "
              f"({keys_in_split[:3]}{'...' if len(keys_in_split)>3 else ''})")
        print(f"Matched        : {len(matched)}")
        if not matched:
            print("\n[ERROR] No lib directories match split.json keys!")
            print("  First 3 dirs on disk :", dirs_on_disk[:3])
            print("  First 3 keys in split:", keys_in_split[:3])
            print("  → Likely cause: split.json was built from a different"
                  " data_dir or the paths inside split.json use a different"
                  " bundler dir name than what is on disk.")

        if mode == "closed":
            # Show a sample key from split.json vs a sample path on disk
            # so the user can spot the mismatch immediately.
            sample_lib = keys_in_split[0] if keys_in_split else None
            if sample_lib and isinstance(lib_split.get(sample_lib), dict):
                sample_split_key = next(iter(lib_split[sample_lib]))
                print(f"  sample split.json key : '{sample_lib}' → "
                      f"'{sample_split_key}'")
            if dirs_on_disk:
                sample_disk_lib = matched[0] if matched else dirs_on_disk[0]
                disk_lib_dir    = osp.join(graph_root, sample_disk_lib)
                for bver in sorted(os.listdir(disk_lib_dir)):
                    gdir = osp.join(disk_lib_dir, bver, "graphs")
                    if osp.isdir(gdir):
                        gfiles = _graph_files(gdir)
                        if gfiles:
                            print(f"  sample disk path      : '{sample_disk_lib}'"
                                  f" → '{bver}/graphs/{gfiles[0]}'")
                        break

        for lib_ver in sorted(os.listdir(graph_root)):
            lib_dir = osp.join(graph_root, lib_ver)
            if not osp.isdir(lib_dir):
                continue
            # ---- lib filter ----
            if self._lib_filter and not _lib_matches(lib_ver, self._lib_filter, spliter=self.spliter):
                log.debug("Skipping lib %s (not in lib_filter)", lib_ver)
                continue
            if lib_ver not in lib_split:
                stats["skip_not_in_split"] += 1
                continue

            lib_idx    = lib_to_idx[lib_ver]
            lib_info   = lib_split[lib_ver]   # str or dict depending on mode

            for bundler_ver in sorted(os.listdir(lib_dir)):
                bundler_dir = osp.join(lib_dir, bundler_ver)
                if not osp.isdir(bundler_dir):
                    continue           # skip bundle.js, build.log, etc.
                # ---- bundler filter ----
                if self._bundler_filter is not None:
                    bname = bundler_ver.split(self.spliter)[0]   # base name
                    if (bundler_ver not in self._bundler_filter
                            and bname not in self._bundler_filter):
                        log.debug("Skipping bundler %s (not in filter)",
                                  bundler_ver)
                        continue
                graphs_dir = osp.join(bundler_dir, "graphs")
                if not osp.isdir(graphs_dir):
                    continue

                bundler_name, bundler_version = _split_bundler_ver(bundler_ver)
                fnames = _graph_files(graphs_dir)
                if self.max_graphs_per_bundler is not None:
                    fnames = fnames[: self.max_graphs_per_bundler]

                for fname in fnames:

                    # ---- resolve & normalise split key for this graph ----
                    if mode == "closed":
                        graph_key = f"{bundler_ver}/graphs/{fname}"
                        if graph_key not in lib_info:
                            continue   # graph not listed in split.json
                        split_key = lib_info[graph_key]
                    else:
                        split_key = lib_info   # plain string from split.json

                    # Always use 'val' (never 'valid') — guard bad values too
                    split_key = "val" if split_key == "valid" else split_key
                    if split_key not in VALID_SPLITS:
                        log.warning("Unknown split value %r for %s/%s -- skipping",
                                    split_key, lib_ver, fname)
                        continue

                    # ---- parse ----
                    fpath = osp.join(graphs_dir, fname)
                    try:
                        G = load_graph(fpath)
                    except Exception as exc:
                        log.warning("Parse error %s/%s/%s: %s",
                                    lib_ver, bundler_ver, fname, exc)
                        stats["skip_parse"] += 1
                        continue

                    # ---- size filter ----
                    n = G.number_of_nodes()
                    if n < self.min_nodes or n > self.max_nodes:
                        stats["skip_size"] += 1
                        continue

                    # ---- convert ----
                    data = nx_to_pyg(G, lib_idx, vocab=vocab)
                    if data is None:
                        stats["skip_empty"] += 1
                        continue

                    # ---- metadata ----
                    data.lib_ver       = lib_ver
                    data.bundler_name  = bundler_name
                    data.bundler_ver   = bundler_version
                    data.graph_id      = fname
                    data.split         = split_key   # convenient for analysis

                    graph_idx = len(data_list)
                    data_list.append(data)
                    split_dict[split_key].append(graph_idx)
                    stats["loaded"] += 1

        # ---- report ----
        print(
            f"Loaded {stats['loaded']} graphs  "
            f"(train={len(split_dict['train'])}  "
            f"val={len(split_dict['val'])}  "
            f"test={len(split_dict['test'])})  "
            f"skipped: parse={stats['skip_parse']}  "
            f"size={stats['skip_size']}  "
            f"empty={stats['skip_empty']}"
        )

        if not data_list:
            raise RuntimeError(
                "No graphs loaded. Check:\n"
                "  1. raw/ contains lib@ver/ subdirectories\n"
                "  2. split.json keys match directory names exactly\n"
                "  3. graphs/ subdirectories contain .xml / .dot files\n"
                f"  4. Split mode detected as '{mode}' — "
                "verify split.json schema matches"
            )

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict,              self.processed_paths[1])

    # ------------------------------------------------------------------ #

    def get_idx_split(self) -> Dict[str, List[int]]:
        """
        Returns {"train": [...], "val": [...], "test": [...]}.
        Always uses 'val' (never 'valid') — safe to call on old processed files.
        """
        d = torch.load(self.processed_paths[1], weights_only=False)
        # backward-compat: rename 'valid' key if an old processed file exists
        if "valid" in d and "val" not in d:
            d["val"] = d.pop("valid")
        elif "valid" in d and "val" in d:
            d["val"] = d["val"] + d.pop("valid")
        # ensure all three keys always present
        for k in ("train", "val", "test"):
            d.setdefault(k, [])
        return d

    def __repr__(self) -> str:
        return f"JSLibsDataset(graphs={len(self)}, classes={self.num_classes})"


# =============================================================================
# Label map  (inference helper)
# =============================================================================

def build_label_map(split_json: str) -> Dict[int, str]:
    """idx → lib@ver  (mirrors JSLibsDataset.process() ordering)."""
    with open(split_json) as f:
        lib_split = json.load(f)
    return {i: lib for i, lib in enumerate(sorted(lib_split.keys()))}


# =============================================================================
# Test / debug
# python graphgps/loader/dataset/jslibs.py
# =============================================================================

if __name__ == "__main__":
    import pprint
    from collections import Counter

    ROOT = "datasets/JSLibs"

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    dataset = JSLibsDataset(
        root=ROOT,
        data_dir="/home/aiuser4/ado/bundled-js-scan/data/train/v2.2",  # ← override raw_dir for graphs
        split_path=osp.join(ROOT, "raw", "split.json"),
        max_graphs_per_bundler=10  # keep small for debugging
    )

    print("\nDataset loaded:")
    print(dataset)
    print(f"Total graphs: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")

    print("\n" + "=" * 60)
    print("Checking splits...")
    print("=" * 60)

    split = dataset.get_idx_split()
    pprint.pprint(split)

    def check_split(name, idxs):
        print(f"\n--- {name.upper()} ---")
        print(f"Size: {len(idxs)}")

        if len(idxs) == 0:
            print("⚠️  EMPTY SPLIT (this will crash training!)")
            return

        ys = [dataset[i].y.item() for i in idxs]
        print("Label distribution:", Counter(ys))

    check_split("train", split["train"])
    check_split("val", split['val'])
    check_split("test", split["test"])

    print("\n" + "=" * 60)
    print("Inspecting sample graph...")
    print("=" * 60)

    data = dataset[0]

    print("Graph info:")
    print(f"- num_nodes: {data.num_nodes}")
    print(f"- num_edges: {data.edge_index.shape[1]}")
    print(f"- node_feat_dim: {data.x.shape}")
    print(f"- edge_attr shape: {data.edge_attr.shape}")

    print("\nNode feature sample (non-zero indices):")
    nz = (data.x[0] > 0).nonzero(as_tuple=True)[0]
    print(nz[:10])

    print("\nEdge attr sample (first 10):")
    print(data.edge_attr[:10])

    print("\nUnique edge groups:")
    print(set(data.edge_attr[:, 0].tolist()))

    print("\nDirection distribution:")
    print(Counter(data.edge_attr[:, 1].tolist()))

    print("\n" + "=" * 60)
    print("Sanity checks")
    print("=" * 60)

    assert len(split["train"]) > 0, "Train split is empty!"
    assert len(split["val"]) > 0, "Val split is empty!"
    assert len(split["test"]) > 0, "Test split is empty!"

    print("✅ All sanity checks passed!")
