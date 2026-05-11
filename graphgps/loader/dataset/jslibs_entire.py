"""
graphgps/loader/dataset/jslibs_entire.py

Load whole-program CPG (_program.xml/_program.dot), extract k-hop ego
subgraphs around function-entry nodes, and convert to PyG Data objects.

Split modes
-----------
closed  {"axios@1.7.9": {"bundler@ver/graphs/_program.xml": "train", ...}}
        → Same lib in train+val+test, different graphs.
        → Subgraphs extracted from the same _program file are randomly
          partitioned into train/val/test via _assign_subgraph_splits().

open    {"axios@1.7.9": "train", ...}
        → Each lib entirely in one split.
        → All subgraphs from the same bundle share the bundle-level split.

Key parameters
--------------
max_depth      : BFS hop radius (analogous to GNN layers)
min_nodes      : drop subgraphs smaller than this
max_nodes      : drop subgraphs larger than this
closed_*_ratio : train/val split fractions for closed mode
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import pydot
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data, InMemoryDataset

log = logging.getLogger(__name__)


def _lib_matches(lib_ver: str, lib_filter: List[str]) -> bool:
    if not lib_filter:
        return True
    base = ("@" + lib_ver.split("@")[1]
            if lib_ver.startswith("@") else lib_ver.split("@")[0])
    return lib_ver in lib_filter or base in lib_filter


EDGE_GROUPS: Dict[str, int] = {
    "AST": 0,        "CONTAINS": 0,
    "CFG": 1,        "DOMINATE": 1,   "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4,       "ARGUMENT": 4,   "PARAMETER_LINK": 4,
    "REF": 5,
}
NUM_EDGE_GROUPS  = len(set(EDGE_GROUPS.values()))
NODE_FEATURE_DIM = 128

FUNCTION_NODE_KEYWORDS: Tuple[str, ...] = (
    "METHOD",
    "FUNCTION",
    "FunctionDeclaration",
    "ArrowFunctionExpression",
    "FunctionExpression",
)

PROGRAM_FILE_STEMS: Tuple[str, ...] = ("_program",)


def detect_split_mode(lib_split: Dict) -> str:
    first = next(iter(lib_split.values()))
    if isinstance(first, dict):
        return "closed"
    if isinstance(first, str):
        return "open"
    raise ValueError(f"Unrecognised split.json schema — expected str or dict, got {type(first)}")


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


def load_program_graph(graphs_dir: str) -> Tuple[Optional[nx.MultiDiGraph], Optional[str]]:
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
                    return None, fname
    return None, None


def _is_function_node(label: str) -> bool:
    label_up = label.upper()
    return any(kw.upper() in label_up for kw in FUNCTION_NODE_KEYWORDS)


def _bfs_neighborhood(
    G: nx.MultiDiGraph,
    root: str,
    max_depth: int,
) -> Set[str]:
    visited: Set[str] = {root}
    queue: deque[Tuple[str, int]] = deque([(root, 0)])
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
    results: List[Tuple[str, nx.MultiDiGraph]] = []

    entry_nodes = [
        node for node, attr in G.nodes(data=True)
        if _is_function_node(attr.get("label", ""))
    ]

    if not entry_nodes:
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


def _encode_nodes(
    G: nx.MultiDiGraph,
    node2id: Dict,
    vocab: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
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
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        num_nodes=len(node2id),
        y=torch.tensor([label], dtype=torch.long),
    )


def _split_bundler_ver(bundler_ver: str) -> Tuple[str, str]:
    if "@" in bundler_ver:
        name, ver = bundler_ver.split("@", 1)
        return name, ver
    return bundler_ver, ""


def _resolve_program_split_key(
    mode: str,
    lib_info,
    bundler_ver: str,
    fname: str,
) -> Optional[str]:
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
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
) -> List[str]:
    rng = random.Random(seed)
    indices = list(range(n_subgraphs))
    rng.shuffle(indices)

    n_train = max(1, int(n_subgraphs * train_ratio))
    n_val   = max(1, int(n_subgraphs * val_ratio))
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


class JSLibsEntireDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        data_dir: Optional[str]               = None,
        split_path: Optional[str]             = None,
        max_depth: int                        = 3,
        min_nodes: int                        = 5,
        max_nodes: int                        = 2000,
        max_graphs_per_bundler: Optional[int] = None,
        bundler_filter: Optional[List[str]]  = None,
        lib_filter: Optional[List[str]]       = None,
        closed_train_ratio: float             = 0.70,
        closed_val_ratio: float               = 0.15,
        transform: Optional[Callable]         = None,
        pre_transform: Optional[Callable]       = None,
        pre_filter: Optional[Callable]        = None,
    ):
        self._data_dir              = data_dir
        self.split_path             = split_path or osp.join(root, "raw", "split.json")
        self.max_depth              = max_depth
        self.min_nodes              = min_nodes
        self.max_nodes              = max_nodes
        self.max_graphs_per_bundler = max_graphs_per_bundler
        self._bundler_filter        = set(bundler_filter) if bundler_filter else None
        self._lib_filter            = list(lib_filter) if lib_filter else None
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

    def process(self):
        vocab_path = osp.join(self.raw_dir, "cpg_vocab.json")
        vocab: Optional[Dict[str, int]] = None
        if osp.exists(vocab_path):
            with open(vocab_path) as f:
                vocab = json.load(f)
            log.info("Loaded CPG vocab: %d entries", len(vocab))
        else:
            log.warning(
                "cpg_vocab.json not found at %s — using hash node features.",
                vocab_path,
            )

        with open(self.split_path) as f:
            lib_split: Dict = json.load(f)

        mode = detect_split_mode(lib_split)
        log.info("Split mode: %s", mode)
        print(f"[jslibs_entire] Split mode: {mode}  |  max_depth={self.max_depth}")

        all_libs   = sorted(lib_split.keys())
        lib_to_idx = {lib: i for i, lib in enumerate(all_libs)}

        VALID_SPLITS = {"train", "val", "test"}

        data_list:  List[Data]      = []
        split_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
        stats = {
            "bundles_seen":       0,
            "bundles_no_prog":    0,
            "bundles_broken":     0,
            "subgraphs_total":   0,
            "subgraphs_loaded":  0,
            "skip_size":         0,
            "skip_empty":        0,
            "skip_not_listed":   0,
            "skip_not_in_split": 0,
        }

        graph_root = self.graph_dir
        print(f"[jslibs_entire] Graph source: {graph_root}")
        if self._lib_filter:
            print(f"[jslibs_entire] Lib filter   : {sorted(self._lib_filter)}")
        else:
            print("[jslibs_entire] Lib filter   : ALL")
        if self._bundler_filter:
            print(f"[jslibs_entire] Bundler filter: {sorted(self._bundler_filter)}")
        else:
            print("[jslibs_entire] Bundler filter: ALL")

        dirs_on_disk  = sorted(d for d in os.listdir(graph_root)
                               if osp.isdir(osp.join(graph_root, d)))
        keys_in_split = sorted(lib_split.keys())
        matched = [d for d in dirs_on_disk if d in lib_split]
        print(f"[jslibs_entire] Libs on disk: {len(dirs_on_disk)}  "
              f"in split: {len(keys_in_split)}  matched: {len(matched)}")

        for lib_ver in sorted(os.listdir(graph_root)):
            lib_dir = osp.join(graph_root, lib_ver)
            if not osp.isdir(lib_dir):
                continue
            if self._lib_filter and not _lib_matches(lib_ver, self._lib_filter):
                log.debug("Skipping lib %s (not in lib_filter)", lib_ver)
                continue
            if lib_ver not in lib_split:
                stats["skip_not_in_split"] += 1
                continue

            lib_idx  = lib_to_idx[lib_ver]
            lib_info = lib_split[lib_ver]

            for bundler_ver in sorted(os.listdir(lib_dir)):
                bundler_dir = osp.join(lib_dir, bundler_ver)
                if not osp.isdir(bundler_dir):
                    continue
                if self._bundler_filter is not None:
                    bname = bundler_ver.split("@")[0]
                    if (bundler_ver not in self._bundler_filter
                            and bname not in self._bundler_filter):
                        log.debug("Skipping bundler %s (not in filter)", bundler_ver)
                        continue
                graphs_dir = osp.join(bundler_dir, "graphs")
                if not osp.isdir(graphs_dir):
                    continue

                stats["bundles_seen"] += 1
                bundler_name, bundler_version = _split_bundler_ver(bundler_ver)

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

                if mode == "open":
                    bundle_split_key = _resolve_program_split_key(
                        mode, lib_info, bundler_ver, prog_fname
                    )
                    if bundle_split_key not in VALID_SPLITS:
                        stats["skip_not_listed"] += 1
                        continue

                subgraphs = extract_function_subgraphs(
                    G,
                    max_depth=self.max_depth,
                    min_nodes=self.min_nodes,
                    max_nodes=self.max_nodes,
                )
                stats["subgraphs_total"] += len(subgraphs)

                if self.max_graphs_per_bundler is not None:
                    subgraphs = subgraphs[: self.max_graphs_per_bundler]

                n = len(subgraphs)
                if n == 0:
                    continue

                if mode == "closed":
                    bundle_seed = lib_idx * 10007 + hash(bundler_ver) % 9973
                    subgraph_splits = _assign_subgraph_splits(
                        n,
                        seed=bundle_seed,
                        train_ratio=self.closed_train_ratio,
                        val_ratio=self.closed_val_ratio,
                    )
                else:
                    subgraph_splits = [bundle_split_key] * n

                for (anchor_id, sub), split_key in zip(subgraphs, subgraph_splits):
                    if split_key not in VALID_SPLITS:
                        continue

                    data = nx_to_pyg(sub, lib_idx, vocab=vocab)
                    if data is None:
                        stats["skip_empty"] += 1
                        continue

                    data.lib_ver      = lib_ver
                    data.bundler_name = bundler_name
                    data.bundler_ver  = bundler_version
                    data.anchor_node  = str(anchor_id)
                    data.split        = split_key

                    graph_idx = len(data_list)
                    data_list.append(data)
                    split_dict[split_key].append(graph_idx)
                    stats["subgraphs_loaded"] += 1

        print(
            f"[jslibs_entire] Loaded {stats['subgraphs_loaded']} subgraphs  "
            f"(train={len(split_dict['train'])}  "
            f"val={len(split_dict['val'])}  "
            f"test={len(split_dict['test'])})\n"
            f"         bundles: seen={stats['bundles_seen']}  "
            f"no_prog={stats['bundles_no_prog']}  "
            f"broken={stats['bundles_broken']}\n"
            f"         subgraphs: total={stats['subgraphs_total']}  "
            f"skip_size={stats['skip_size']}  "
            f"skip_empty={stats['skip_empty']}"
        )

        if not data_list:
            raise RuntimeError(
                "No subgraphs loaded. Check:\n"
                "  1. <graph_dir>/lib@ver/bundler@ver/graphs/_program.xml exists\n"
                "  2. split.json keys match directory names\n"
                "  3. max_depth/min_nodes/max_nodes filters"
            )

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict,              self.processed_paths[1])

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


def build_label_map(split_json: str) -> Dict[int, str]:
    with open(split_json) as f:
        lib_split = json.load(f)
    return {i: lib for i, lib in enumerate(sorted(lib_split.keys()))}


if __name__ == "__main__":
    import pprint
    from collections import Counter

    ROOT = "datasets/JSLibs"

    print("=" * 60)
    print("Loading dataset (jslibs_entire mode)...")
    print("=" * 60)

    dataset = JSLibsDataset(
        root=ROOT,
        data_dir="/home/aiuser4/ado/bundled-js-scan/data/train/v2.2",
        split_path=osp.join(ROOT, "raw", "split.json"),
        max_depth=3,
        min_nodes=5,
        max_nodes=500,
        max_graphs_per_bundler=20,
    )

    print(dataset)
    print(f"Total subgraphs: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")

    split = dataset.get_idx_split()
    print("\nSplit sizes:")
    pprint.pprint({k: len(v) for k, v in split.items()})

    for name, idxs in split.items():
        if not idxs:
            print(f"EMPTY {name.upper()} split!")
            continue
        ys = [dataset[i].y.item() for i in idxs]
        print(f"\n{name}: label distribution (top 10)")
        pprint.pprint(Counter(ys).most_common(10))

    d = dataset[0]
    print(f"\nSample[0]: lib_ver={d.lib_ver} anchor={d.anchor_node} split={d.split}")
    print(f"num_nodes={d.num_nodes} num_edges={d.edge_index.shape[1]}")

    assert len(split["train"]) > 0, "Train empty!"
    assert len(split["val"])   > 0, "Val empty!"
    assert len(split["test"])  > 0, "Test empty!"
    print("\n All sanity checks passed")
