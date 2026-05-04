import json
import logging
import os
import os.path as osp
from typing import Dict, Callable, List, Optional, Tuple
 
import torch
import networkx as nx
import pydot
from torch_geometric.data import Data, InMemoryDataset
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple
 
 
log = logging.getLogger(__name__)
 
# ---- Edge type mapping (GLOBAL FIXED) ----
EDGE_TYPE_MAP = {
    "AST": 0,
    "CFG": 1,
    "REACHING_DEF": 2,  # PDG
    "CDG": 3,
}
 
EDGE_GROUPS = {
    # ---- Syntax ----
    "AST": 0,
    "CONTAINS": 0,
 
    # ---- Control Flow ----
    "CFG": 1,
    "DOMINATE": 1,
    "POST_DOMINATE": 1,
 
    # ---- Data Flow ----
    "REACHING_DEF": 2,
 
    # ---- Control Dependence ----
    "CDG": 3,
 
    # ---- Call / Argument ----
    "CALL": 4,
    "ARGUMENT": 4,
    "PARAMETER_LINK": 4,
 
    # ---- Reference ----
    "REF": 5,
}
 
 
# =========================
# DOT → PyG utils
# =========================
def read_dot(path):
    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise ValueError(f"Cannot parse DOT: {path}")
 
    P = graphs[0]
 
    # Fix compatibility issue
    try:
        G = nx.drawing.nx_pydot.from_pydot(P)
    except TypeError:
        # fallback manual conversion
        G = nx.MultiDiGraph()
 
        for node in P.get_nodes():
            G.add_node(node.get_name(), **node.get_attributes())
 
        for edge in P.get_edges():
            G.add_edge(
                edge.get_source(),
                edge.get_destination(),
                **edge.get_attributes()
            )
 
    return G
 
 
def read_dot_safe(path):
    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise ValueError(f"Cannot parse DOT: {path}")
 
    P = graphs[0]
 
    # Tự build graph → tránh bug networkx-pydot
    G = nx.MultiDiGraph()
 
    # ---- Nodes ----
    for node in P.get_nodes():
        name = node.get_name()
 
        # Skip node ảo của pydot
        if name in ("node", "graph", "edge"):
            continue
 
        name = name.strip('"')
        attrs = {k: v.strip('"') for k, v in node.get_attributes().items()}
        G.add_node(name, **attrs)
 
    # ---- Edges ----
    for edge in P.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        attrs = {k: v.strip('"') for k, v in edge.get_attributes().items()}
        G.add_edge(src, dst, **attrs)
 
    return G
 
 
def read_xml_safe(path):
    # ---- Try GraphML ----
    try:
        G = nx.read_graphml(path)
        return nx.MultiDiGraph(G)
    except Exception:
        pass
 
    # ---- Try GEXF ----
    try:
        G = nx.read_gexf(path)
        return nx.MultiDiGraph(G)
    except Exception:
        pass
 
    # ---- Custom XML fallback ----
    G = nx.MultiDiGraph()
 
    tree = ET.parse(path)
    root = tree.getroot()
 
    # heuristic: find nodes
    for node in root.findall(".//node"):
        nid = node.get("id")
        if nid is None:
            continue
 
        attrs = {}
        for k, v in node.attrib.items():
            attrs[k] = v
 
        # parse nested <data key="label">
        for data in node.findall(".//data"):
            key = data.get("key")
            val = data.text
            if key and val:
                attrs[key] = val
 
        G.add_node(nid, **attrs)
 
    # heuristic: find edges
    for edge in root.findall(".//edge"):
        src = edge.get("source")
        dst = edge.get("target")
 
        if src is None or dst is None:
            continue
 
        attrs = {}
        for k, v in edge.attrib.items():
            attrs[k] = v
 
        for data in edge.findall(".//data"):
            key = data.get("key")
            val = data.text
            if key and val:
                attrs[key] = val
 
        G.add_edge(src, dst, **attrs)
 
    return G
 
 
def encode_edges_grouped(G, node2id):
    edge_index = []
    edge_attr = []
 
    for u, v, attr in G.edges(data=True):
        if u not in node2id or v not in node2id:
            continue
 
        src = node2id[u]
        dst = node2id[v]
 
        etype = (
            attr.get("label") or
            attr.get("type") or
            "AST"
        )
 
        group = EDGE_GROUPS.get(etype, 0)
 
        # ---- Forward edge ----
        edge_index.append([src, dst])
        edge_attr.append([group, 0])  # 0 = forward
 
        # ---- Reverse edge (VERY IMPORTANT for GNN) ----
        edge_index.append([dst, src])
        edge_attr.append([group, 1])  # 1 = reverse
 
    if len(edge_index) == 0:
        return None, None
 
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
 
    return edge_index, edge_attr
 
 
# =========================
# Helper
# =========================
def _load_graph_from_file(fpath: str):
    """
    Load a single .dot or .xml file → nx.MultiDiGraph.
    Mirrors the logic in jslibs.py so features are identical to training.
    """
    import networkx as nx
    import pydot
    import xml.etree.ElementTree as ET
 
    fname = osp.basename(fpath)
 
    if fname.endswith(".dot"):
        graphs = pydot.graph_from_dot_file(fpath)
        if not graphs:
            raise ValueError(f"Cannot parse DOT: {fpath}")
        P = graphs[0]
        G = nx.MultiDiGraph()
        for node in P.get_nodes():
            name = node.get_name()
            if name in ("node", "graph", "edge"):
                continue
            name = name.strip('"')
            attrs = {k: v.strip('"') for k, v in node.get_attributes().items()}
            G.add_node(name, **attrs)
        for edge in P.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            attrs = {k: v.strip('"') for k, v in edge.get_attributes().items()}
            G.add_edge(src, dst, **attrs)
 
    elif fname.endswith(".xml"):
        try:
            G = nx.read_graphml(fpath)
            G = nx.MultiDiGraph(G)
        except Exception:
            try:
                G = nx.read_gexf(fpath)
                G = nx.MultiDiGraph(G)
            except Exception:
                G = nx.MultiDiGraph()
                tree = ET.parse(fpath)
                root = tree.getroot()
                for node in root.findall(".//node"):
                    nid = node.get("id")
                    if nid is None:
                        continue
                    attrs = dict(node.attrib)
                    for data in node.findall(".//data"):
                        k, v = data.get("key"), data.text
                        if k and v:
                            attrs[k] = v
                    G.add_node(nid, **attrs)
                for edge in root.findall(".//edge"):
                    src, dst = edge.get("source"), edge.get("target")
                    if src is None or dst is None:
                        continue
                    attrs = dict(edge.attrib)
                    for data in edge.findall(".//data"):
                        k, v = data.get("key"), data.text
                        if k and v:
                            attrs[k] = v
                    G.add_edge(src, dst, **attrs)
    else:
        raise ValueError(f"Unsupported file type: {fpath}")
 
    return G
 
 
def nx_graph_to_pyg(G) -> Optional[Data]:
    """Convert nx.MultiDiGraph → torch_geometric.data.Data (no label)."""
    if G.number_of_nodes() == 0:
        return None
 
    node2id = {n: i for i, n in enumerate(G.nodes())}
    dim = 128
    x = torch.zeros((len(node2id), dim))
 
    for node, attr in G.nodes(data=True):
        idx = node2id[node]
        label = attr.get("label", "UNK")
        h = hash(label) % dim
        x[idx][h] = 1.0
 
    edge_index, edge_attr = [], []
    for u, v, attr in G.edges(data=True):
        if u not in node2id or v not in node2id:
            continue
        src, dst = node2id[u], node2id[v]
        etype = attr.get("label") or attr.get("type") or "AST"
        group = EDGE_GROUPS.get(etype, 0)
        edge_index.append([src, dst]);  edge_attr.append([group, 0])
        edge_index.append([dst, src]);  edge_attr.append([group, 1])
 
    if not edge_index:
        return None
 
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        num_nodes=len(node2id),
    )
 
 
def load_graphs_from_dir(graphs_dir: str) -> List[Tuple[str, Data]]:
    """
    Load all .dot/.xml files from a directory.
    Returns list of (filename, Data) pairs.
    """
    results = []
    for fname in sorted(os.listdir(graphs_dir)):
        if "Zone.Identifier" in fname or fname.startswith("_program"):
            continue
        if not (fname.endswith(".dot") or fname.endswith(".xml")):
            continue
        fpath = osp.join(graphs_dir, fname)
        try:
            G = _load_graph_from_file(fpath)
            data = nx_graph_to_pyg(G)
            if data is not None:
                results.append((fname, data))
        except Exception as e:
            log.warning(f"Skipped {fname}: {e}")
    return results
 

# =========================
# Label map builder (for JSLibs)
# =========================
def build_label_map(split_json: str) -> Dict[int, str]:
    """
    Reproduces the lib_to_idx mapping from JSLibsDataset.process().
    Returns idx → lib_name dict.
    """
    with open(split_json) as f:
        lib_split = json.load(f)
    all_libs = sorted(lib_split.keys())
    return {i: lib for i, lib in enumerate(all_libs)}


# =========================
# Dataset Loader
 
# datasets/JSLibs/
#  ├── raw/
#  └── processed/
#       ├── data.pt
#       └── split_dict.pt
 
# =========================
class JSLibsDataset(InMemoryDataset):
 
    def __init__(
        self,
        root: str,
        split_path: Optional[str] = None,
        min_nodes: int = 5,
        max_nodes: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.split_path = split_path or osp.join(root, 'raw', 'split.json')
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
 
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)
 
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')
 
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
 
    @property
    def raw_file_names(self):
        return ['split.json']
 
    @property
    def processed_file_names(self):
        return ['data.pt', 'split_dict.pt']
   
    @property
    def num_classes(self):
        if hasattr(self.data, 'y') and self.data.y is not None:
            return int(self.data.y.max().item() + 1)
        return 0
 
    def download(self):
        pass
 
    def process(self):
        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}
 
        # ---- Load split ----
        with open(self.split_path) as f:
            lib_split = json.load(f)
 
        all_libs = sorted(lib_split.keys())
        lib_to_idx = {lib: i for i, lib in enumerate(all_libs)}
 
        skipped = 0
 
        for lib in os.listdir(self.raw_dir):
            lib_dir = osp.join(self.raw_dir, lib)
            if not osp.isdir(lib_dir) or lib not in lib_split:
                continue
 
            split = lib_split[lib]
            split_key = 'valid' if split == 'val' else split
            lib_idx = lib_to_idx[lib]
 
            for bundler in os.listdir(lib_dir):
                # skip non-directory files
                if not osp.isdir( osp.join(lib_dir, bundler)):
                    continue
 
                bundler_dir = osp.join(lib_dir, bundler)
                graphs_dir = osp.join(bundler_dir, 'graphs')
 
                if not osp.isdir(graphs_dir):
                    continue
 
                for fname in os.listdir(graphs_dir)[:10]: # for testing, only load x graphs per bundler
                    # skip Windows artifact
                    if "Zone.Identifier" in fname:
                        continue
                    # skip _program.dot
                    if fname.startswith("_program"):
                        continue
 
                    fpath = osp.join(graphs_dir, fname)
 
                    try:
                        if fname.endswith(".dot"):
                            G = read_dot_safe(fpath)
                        elif fname.endswith(".xml"):
                            G = read_xml_safe(fpath)
                        else:
                            skipped += 1
                            continue
                    except Exception as e:
                        skipped += 1
                        continue
 
                    node2id = {n: i for i, n in enumerate(G.nodes())}
 
                    # ---- Node features (HASH, không vocab) ----
                    dim = 128
                    x = torch.zeros((len(node2id), dim))
 
                    for node, attr in G.nodes(data=True):
                        idx = node2id[node]
                        label = attr.get("label", "UNK")
                        h = hash(label) % dim
                        x[idx][h] = 1
 
                    # ---- Edges ----
                    # edge_index = []
                    # edge_attr = []
 
                    # for u, v, attr in G.edges(data=True):
                    #     edge_index.append([node2id[u], node2id[v]])
 
                    #     etype = attr.get("label", "AST")
                    #     edge_attr.append(EDGE_TYPE_MAP.get(etype, 0))
 
                    # if len(edge_index) == 0:
                    #     skipped += 1
                    #     continue
 
                    # edge_index = torch.tensor(edge_index).t().contiguous()
                    # edge_attr = torch.tensor(edge_attr).view(-1, 1)
 
                    edge_index, edge_attr = encode_edges_grouped(G, node2id)
 
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        num_nodes=len(node2id),
                        y=torch.tensor([lib_idx])
                    )
 
                    # ---- Metadata ----
                    data.lib = lib
                    data.bundler = bundler
                    data.graph_id = fname
                    data.y = lib_idx # label = lib index
                    # data.bundler_id = bundler_idx
 
                    data_list.append(data)
                    idx = len(data_list) - 1
                    split_dict[split_key].append(idx)
 
                    print(
                        "Read %10s:%25s with %5s nodes and %5s edges." % 
                        (lib, fname, G.number_of_nodes(), G.number_of_edges())
                    )
 
        print(f"Loaded {len(data_list)} graphs, skipped {skipped}")
 
        if len(data_list) == 0:
            raise RuntimeError("No graphs loaded")
 
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
 
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
 
        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])
 
 
    def get_idx_split(self):
        return torch.load(self.processed_paths[1], weights_only=False)
 
    def __repr__(self):
        return f"JSLibDataset(graphs={len(self)})"''
 
 
if __name__ == '__main__':
    dataset = JSLibsDataset(root='datasets/JSLibs')
    # print(dataset)
    # print(dataset[0])
    # print(dataset[0].x)
    # print(dataset[0].edge_index)
    # print(dataset[0].edge_attr)
    # print(dataset.get_idx_split())

    from collections import Counter

    split = dataset.get_idx_split()

    def check_split(name, idxs):
        ys = [dataset[i].y.item() for i in idxs]
        print(f"{name} size:", len(idxs))
        print(f"{name} label dist:", Counter(ys))

    check_split("Train", split['train'])
    check_split("Valid", split['valid'])
    check_split("Test", split['test'])
 