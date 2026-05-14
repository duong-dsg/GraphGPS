"""
graphgps/inference/prototype_inference.py

Prototype-based inference for OSS library detection from bundled JS code.

Pipeline:
  1. Load model checkpoint + class prototypes
  2. Parse input graphs (either whole-program CPG or pre-extracted subgraphs)
  3. Run each subgraph through encoder to get embeddings
  4. Compute cosine similarity to class prototypes
  5. Aggregate per-graph scores → per-library confidence

load_type:
  - 'entire'   : Load _program.xml/.dot, extract k-hop function subgraphs via BFS
  - 'individual': Load pre-extracted subgraph files directly

Usage:
  >>> # Load entire program CPGs and extract subgraphs
  >>> infer = PrototypeInference(
  ...     model_path="results/run1/model.pt",
  ...     prototypes_path="results/run1/jslibs_prototypes.pt",
  ...     load_type="entire",
  ... )
  >>> results = infer.predict("path/to/bundle/graphs/_program.xml")

  >>> # Load pre-extracted individual subgraph files
  >>> infer = PrototypeInference(
  ...     model_path="results/run1/model.pt",
  ...     prototypes_path="results/run1/jslibs_prototypes.pt",
  ...     load_type="individual",
  ... )
  >>> results = infer.predict("path/to/extracted/subgraphs/")
"""

import json
import logging
import os
import os.path as osp
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader

log = logging.getLogger(__name__)


EDGE_GROUPS: Dict[str, int] = {
    "AST": 0,        "CONTAINS": 0,
    "CFG": 1,        "DOMINATE": 1,   "POST_DOMINATE": 1,
    "REACHING_DEF": 2,
    "CDG": 3,
    "CALL": 4,       "ARGUMENT": 4,   "PARAMETER_LINK": 4,
    "REF": 5,
}
NODE_FEATURE_DIM = 128

FUNCTION_NODE_KEYWORDS: Tuple[str, ...] = (
    "METHOD",
    "FUNCTION",
    "FunctionDeclaration",
    "ArrowFunctionExpression",
    "FunctionExpression",
)
PROGRAM_FILE_STEMS: Tuple[str, ...] = ("_program",)


class PrototypeInference:
    def __init__(
        self,
        model_path: str,
        prototypes_path: str,
        split_json: Optional[str] = None,
        vocab_path: Optional[str] = None,
        load_type: str = "entire",
        max_depth: int = 3,
        min_nodes: int = 5,
        max_nodes: int = 2000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if load_type not in ("individual", "entire"):
            raise ValueError(f"load_type must be 'individual' or 'entire', got '{load_type}'")
        self.load_type = load_type
        self.max_depth = max_depth
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.device = torch.device(device)

        self.model = self._load_model(model_path)

        if prototypes_path.endswith(".pt"):
            self.prototypes, self.label_map = self._load_prototypes(prototypes_path)
        else:
            raise ValueError(
                f"prototypes_path must be a .pt file, got: {prototypes_path}. "
                "Use compute_prototypes_from_data() to create prototypes from processed data."
            )
        self.num_classes = self.prototypes.shape[0]

        if not self.label_map and split_json:
            log.warning("No label_map in prototypes.pt, falling back to split_json")
            self.label_map = self._build_label_map(split_json)

        self.vocab = self._load_vocab(vocab_path or self._find_vocab(split_json))

    def _load_model(self, model_path: str):
        log.info("Loading model from %s", model_path)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model = ckpt.get("model", ckpt)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _load_prototypes(self, prototypes_path: str) -> Tuple[torch.Tensor, Dict[int, str]]:
        log.info("Loading prototypes from %s", prototypes_path)
        data = torch.load(prototypes_path, map_location=self.device, weights_only=False)
        prototypes = data["prototypes"]
        if prototypes.device != self.device:
            prototypes = prototypes.to(self.device)
        label_map = data.get("label_map", {})
        if isinstance(label_map, dict):
            if label_map and not isinstance(list(label_map.keys())[0], int):
                label_map = {int(k): v for k, v in label_map.items()}
        return prototypes, label_map

    def _build_label_map(self, split_json: str) -> Dict[int, str]:
        with open(split_json) as f:
            lib_split = json.load(f)
        all_libs = sorted(lib_split.keys())
        if len(all_libs) == self.num_classes:
            return {i: lib for i, lib in enumerate(all_libs)}
        return self.label_map

    def _find_vocab(self, split_json: Optional[str]) -> Optional[str]:
        if split_json:
            vocab = osp.join(osp.dirname(split_json), "cpg_vocab.json")
            if osp.exists(vocab):
                return vocab
        return None

    def _load_vocab(self, vocab_path: Optional[str]) -> Optional[Dict[str, int]]:
        if vocab_path and osp.exists(vocab_path):
            with open(vocab_path) as f:
                return json.load(f)
        return None

    def predict(
        self,
        input_path: str,
        topk: int = 5,
        topk_per_graph: int = 3,
        return_subgraphs: bool = False,
    ) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
        """
        Run inference on a program graph.

        Args:
            input_path: Path to graph file(s) or directory containing them.
                - For load_type='entire': path to _program.xml/.dot or directory
                  containing lib@ver/bundler@ver/graphs/_program.xml
                - For load_type='individual': path to directory containing
                  pre-extracted subgraph files (func_001.xml, etc.)
            topk: Number of top predictions to return
            topk_per_graph: Top-k similarities per subgraph to aggregate
            return_subgraphs: If True, also return subgraph details

        Returns:
            Dict mapping bundle_name -> list of predictions
            Each prediction: {'rank', 'lib', 'score', 'confidence'}
        """
        if self.load_type == "individual":
            return self._predict_individual(input_path, topk, topk_per_graph, return_subgraphs)
        else:
            return self._predict_entire(input_path, topk, topk_per_graph, return_subgraphs)

    def _predict_individual(
        self,
        input_path: str,
        topk: int,
        topk_per_graph: int,
        return_subgraphs: bool,
    ):
        subgraph_files = self._find_individual_files(input_path)
        if not subgraph_files:
            raise ValueError(f"No subgraph files found at {input_path}")

        bundle_name = osp.basename(input_path.rstrip("/"))
        log.info("Processing %d individual subgraph files from: %s", len(subgraph_files), input_path)

        subgraphs = []
        subgraph_info = []
        for fname, fpath in subgraph_files:
            G = self._load_graph_file(fpath)
            if G is not None and self.min_nodes <= G.number_of_nodes() <= self.max_nodes:
                subgraphs.append((fname, G))
                subgraph_info.append({
                    "file": fname,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                })

        if not subgraphs:
            raise ValueError(f"No valid subgraphs in {input_path}")

        embeddings, _ = self._extract_embeddings(subgraphs)
        similarity = self._cosine_similarity(embeddings, self.prototypes)
        agg_scores = self._aggregate_scores(similarity, topk_per_graph)
        results = self._format_results(agg_scores, topk)

        all_results = {bundle_name: results}
        all_subgraph_details = {bundle_name: subgraph_info} if return_subgraphs else {}

        if return_subgraphs:
            return all_results, all_subgraph_details
        return all_results

    def _predict_entire(
        self,
        input_path: str,
        topk: int,
        topk_per_graph: int,
        return_subgraphs: bool,
    ):
        program_files = self._find_program_files(input_path)
        if not program_files:
            raise ValueError(f"No _program files found at {input_path}")

        all_results = {}
        all_subgraph_details = {}

        for bundle_name, graphs_dir in program_files:
            log.info("Processing: %s", bundle_name)

            G, _ = self._load_program_graph(graphs_dir)
            if G is None:
                log.warning("Failed to load graph for %s", bundle_name)
                continue

            subgraphs = self._extract_function_subgraphs(G)
            if not subgraphs:
                log.warning("No valid subgraphs for %s", bundle_name)
                continue

            embeddings, subgraph_info = self._extract_embeddings(subgraphs)

            similarity = self._cosine_similarity(embeddings, self.prototypes)
            agg_scores = self._aggregate_scores(similarity, topk_per_graph)

            results = self._format_results(agg_scores, topk)
            all_results[bundle_name] = results
            all_subgraph_details[bundle_name] = subgraph_info

        if return_subgraphs:
            return all_results, all_subgraph_details
        return all_results

    def _find_individual_files(self, input_path: str) -> List[Tuple[str, str]]:
        """
        Find pre-extracted individual subgraph files (func_001.xml, etc.)
        for load_type='individual'.

        Expected structure:
            folder/
                graphs/
                    func001.xml
                    func002.xml

        Returns list of (filename, filepath) tuples.
        """
        results = []

        if osp.isfile(input_path):
            if "_program" not in osp.basename(input_path):
                results.append((osp.basename(input_path), input_path))
            return results

        if not osp.isdir(input_path):
            return results

        graphs_dir = osp.join(input_path, "graphs")
        if osp.isdir(graphs_dir):
            input_path = graphs_dir

        for entry in os.listdir(input_path):
            entry_path = osp.join(input_path, entry)
            if osp.isfile(entry_path) and entry.endswith((".xml", ".dot")) and "_program" not in entry and "Zone.Identifier" not in entry:
                results.append((entry, entry_path))

        return sorted(results)

    def _load_graph_file(self, fpath: str):
        """Load a single graph file (.xml or .dot)."""
        try:
            if fpath.endswith(".dot"):
                return self._read_dot(fpath)
            else:
                return self._read_xml(fpath)
        except Exception as exc:
            log.warning("Failed to load %s: %s", fpath, exc)
            return None

    def _find_program_files(self, input_path: str) -> List[Tuple[str, str]]:
        results = []

        if osp.isfile(input_path):
            if "_program" in osp.basename(input_path):
                parent = osp.dirname(input_path)
                bundle_name = osp.basename(osp.dirname(parent))
                results.append((bundle_name, parent))
            return results

        if not osp.isdir(input_path):
            return results

        for entry in os.listdir(input_path):
            entry_path = osp.join(input_path, entry)
            if not osp.isdir(entry_path):
                continue
            graphs_dir = osp.join(entry_path, "graphs")
            for stem in PROGRAM_FILE_STEMS:
                for ext in (".xml", ".dot"):
                    if osp.exists(osp.join(graphs_dir, stem + ext)):
                        results.append((entry, graphs_dir))
                        break
                else:
                    continue
                break
        return results

    def _load_program_graph(self, graphs_dir: str) -> Tuple[Optional[any], Optional[str]]:
        for stem in PROGRAM_FILE_STEMS:
            for ext in (".xml", ".dot"):
                fpath = osp.join(graphs_dir, stem + ext)
                if osp.isfile(fpath):
                    G = self._load_graph_file(fpath)
                    if G is not None:
                        return G, stem + ext
        return None, None

    def _read_dot(self, path: str):
        import pydot
        graphs = pydot.graph_from_dot_file(path)
        if not graphs:
            return None
        P = graphs[0]
        import networkx as nx
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
        return G

    def _read_xml(self, path: str):
        import networkx as nx
        import xml.etree.ElementTree as ET
        for reader in (nx.read_graphml, nx.read_gexf):
            try:
                return nx.MultiDiGraph(reader(path))
            except Exception:
                pass
        G = nx.MultiDiGraph()
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

    def _is_function_node(self, label: str) -> bool:
        label_up = label.upper()
        return any(kw.upper() in label_up for kw in FUNCTION_NODE_KEYWORDS)

    def _bfs_neighborhood(self, G, root: str, max_depth: int) -> Set[str]:
        visited = {root}
        queue = deque([(root, 0)])
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

    def _extract_function_subgraphs(self, G) -> List[Tuple[str, any]]:
        results = []
        entry_nodes = [
            node for node, attr in G.nodes(data=True)
            if self._is_function_node(attr.get("label", ""))
        ]
        if not entry_nodes:
            log.debug("No function-entry nodes — using whole graph")
            if self.min_nodes <= G.number_of_nodes() <= self.max_nodes:
                results.append(("__whole__", G))
            return results

        for anchor in entry_nodes:
            neighborhood = self._bfs_neighborhood(G, anchor, self.max_depth)
            n = len(neighborhood)
            if n < self.min_nodes or n > self.max_nodes:
                continue
            sub = G.subgraph(neighborhood).copy()
            results.append((anchor, sub))
        return results

    def _nx_to_pyg(self, G) -> Optional[Data]:
        if G.number_of_nodes() == 0:
            return None
        node2id = {n: i for i, n in enumerate(G.nodes())}

        if self.vocab is not None:
            indices = [
                self.vocab.get(attr.get("label", "UNK"), 0)
                for _, attr in G.nodes(data=True)
            ]
            x = torch.tensor(indices, dtype=torch.long).unsqueeze(1)
        else:
            x = torch.zeros((len(node2id), NODE_FEATURE_DIM))
            for node, attr in G.nodes(data=True):
                h = hash(attr.get("label", "UNK")) % NODE_FEATURE_DIM
                x[node2id[node]][h] = 1.0

        ei, ea = [], []
        for u, v, attr in G.edges(data=True):
            if u not in node2id or v not in node2id:
                continue
            src = node2id[u]; dst = node2id[v]
            grp = EDGE_GROUPS.get(attr.get("label") or attr.get("type") or "AST", 0)
            ei += [[src, dst], [dst, src]]
            ea += [[grp, 0], [grp, 1]]

        if not ei:
            return None

        return Data(
            x=x,
            edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(ea, dtype=torch.long),
            num_nodes=len(node2id),
        )

    def _extract_embeddings(self, subgraphs, batch_size=32):
        pyg_graphs = []
        subgraph_info = []

        for anchor_id, G in subgraphs:
            data = self._nx_to_pyg(G)
            if data is not None:
                data.anchor_id = str(anchor_id)
                pyg_graphs.append(data)
                subgraph_info.append({
                    "anchor": str(anchor_id),
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                })

        if not pyg_graphs:
            return torch.empty(0, self.num_classes), []

        loader = DataLoader(pyg_graphs, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                if isinstance(out, (tuple, list)):
                    embeddings = out[0]
                else:
                    embeddings = out
                all_embeddings.append(embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings, subgraph_info

    def _cosine_similarity(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(embeddings, prototypes)

    def _aggregate_scores(self, similarity: torch.Tensor, topk_per_graph: int) -> torch.Tensor:
        n_graphs, n_classes = similarity.shape
        agg = torch.zeros(n_classes)

        for i in range(n_graphs):
            row = similarity[i]
            topk = torch.topk(row, k=min(topk_per_graph, n_classes))
            for score, idx in zip(topk.values, topk.indices):
                agg[idx] += score.item()

        return agg

    def _format_results(self, agg_scores: torch.Tensor, topk: int) -> List[Dict]:
        n = min(topk, self.num_classes)
        top_scores, top_indices = torch.topk(agg_scores, k=n)
        total = top_scores.sum().item()

        results = []
        for rank, (score, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist()), start=1
        ):
            lib_name = self.label_map.get(idx, f"class_{idx}")
            confidence = (score / total * 100) if total > 0 else 0
            results.append({
                "rank": rank,
                "lib": lib_name,
                "score": round(score, 4),
                "confidence": round(confidence, 2),
            })
        return results

    def print_results(self, results: Dict[str, List[Dict]], title: str = "OSS Library Detection"):
        for bundle_name, preds in results.items():
            print(f"\n{'='*60}")
            print(f"  {title} — {bundle_name}")
            print(f"{'='*60}")
            print(f"  {'Rank':<5} {'Library':<30} {'Confidence':>12}")
            print(f"  {'-'*60}")
            for p in preds:
                print(f"  {p['rank']:<5} {p['lib']:<30} {p['confidence']:>10.1f}%")
        print(f"{'='*60}\n")

    def save_results(self, results: Dict[str, List[Dict]], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="JSLibs prototype-based inference")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to graph file(s) or directory")
    parser.add_argument("--model", "-m", required=True, help="Model checkpoint path")
    parser.add_argument("--prototypes", "-p", required=True, help="Prototypes path")
    parser.add_argument("--split_json", "-s", default=None, help="split.json path")
    parser.add_argument("--vocab", "-v", default=None, help="cpg_vocab.json path")
    parser.add_argument("--load_type", "-l", default="entire",
                        choices=["individual", "entire"],
                        help="'entire' = load _program.xml and extract subgraphs; "
                             "'individual' = load pre-extracted subgraph files directly")
    parser.add_argument("--max_depth", "-d", type=int, default=3)
    parser.add_argument("--min_nodes", type=int, default=5)
    parser.add_argument("--max_nodes", type=int, default=2000)
    parser.add_argument("--topk", "-k", type=int, default=5)
    parser.add_argument("--topk_per_graph", type=int, default=3)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    infer = PrototypeInference(
        model_path=args.model,
        prototypes_path=args.prototypes,
        split_json=args.split_json,
        vocab_path=args.vocab,
        load_type=args.load_type,
        max_depth=args.max_depth,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        device=args.device,
    )

    results = infer.predict(
        args.input,
        topk=args.topk,
        topk_per_graph=args.topk_per_graph,
    )
    infer.print_results(results)

    if args.output:
        infer.save_results(results, args.output)


def compute_prototypes_from_data(
    data_path: str,
    split_dict_path: str,
    model_path: str,
    output_path: str,
    split_json: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Compute prototypes from processed PyG data + trained model.

    Args:
        data_path: Path to data.pt (contains (Data, dict) tuple from PyG InMemoryDataset)
        split_dict_path: Path to split_dict.pt (contains train/val/test indices)
        model_path: Path to model checkpoint (.pt)
        output_path: Path to save prototypes.pt
        split_json: Path to split.json for label_map (auto-detected if not provided)
        device: Device to run on

    Data format:
        data[0] = Data with concatenated graphs (data[0].x=[total_nodes, 128], data[0].edge_index=[2, total_edges])
        data[1] = dict with slices for each graph + metadata (y, split, graph_id, etc.)
    """
    log.info("Loading data from %s", data_path)
    data_obj = torch.load(data_path, map_location=device, weights_only=False)

    if not isinstance(data_obj, tuple) or len(data_obj) != 2:
        raise ValueError(f"data_path must contain (Data, dict) tuple, got {type(data_obj)}")

    data_full = data_obj[0]
    data_dict = data_obj[1]

    num_graphs = len(data_dict['y'])
    labels = data_dict['y'].cpu().numpy()
    log.info("Data format: (Data, dict) with %d graphs", num_graphs)
    log.info("  data[0].x shape: %s, edge_index shape: %s", data_full.x.shape, data_full.edge_index.shape)

    log.info("Loading split dict from %s", split_dict_path)
    split_dict = torch.load(split_dict_path, map_location='cpu', weights_only=False)

    if isinstance(split_dict, dict) and 'train' in split_dict:
        train_indices = split_dict['train']
        if isinstance(train_indices, torch.Tensor):
            train_indices = train_indices.cpu().numpy()
    else:
        raise ValueError(f"split_dict missing 'train' key. Keys: {split_dict.keys()}")

    if split_json is None:
        split_json = osp.join(osp.dirname(data_path), "..", "raw", "split.json")
    if osp.exists(split_json):
        with open(split_json) as f:
            lib_split = json.load(f)
        all_libs = sorted(lib_split.keys())
        label_map = {i: lib for i, lib in enumerate(all_libs)}
        log.info("Loaded label_map from split.json: %d classes", len(label_map))
    else:
        num_classes = int(labels.max()) + 1
        label_map = {i: str(i) for i in range(num_classes)}
        log.warning("split.json not found at %s, using numeric label_map", split_json)

    num_classes = len(label_map)
    log.info("Loading model from %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt.get("model", ckpt)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    num_classes = len(label_map)
    embedding_dim = 128

    if hasattr(model, "model") and hasattr(model.model, "dim_inner"):
        embedding_dim = model.model.dim_inner
        log.info("Detected embedding_dim from model: %d", embedding_dim)
    elif hasattr(model, "dim_inner"):
        embedding_dim = model.dim_inner
        log.info("Detected embedding_dim from model: %d", embedding_dim)
    else:
        log.warning("Could not detect embedding_dim, using default: %d", embedding_dim)

    log.info("Computing prototypes from %d training samples, %d classes",
             len(train_indices), num_classes)

    support_embeddings: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]

    x_full = data_full.x
    edge_index_full = data_full.edge_index
    edge_attr_full = data_full.edge_attr

    class JSLibsDataset(torch.utils.data.Dataset):
        def __init__(self, data_full, data_dict, indices):
            self.data_full = data_full
            self.data_dict = data_dict
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            graph_idx = self.indices[idx]
            label = int(self.data_dict['y'][graph_idx].item())
            node_start = int(self.data_dict['x'][graph_idx].item())
            node_end = int(self.data_dict['x'][graph_idx + 1].item()) if graph_idx + 1 < len(self.data_dict['x']) else len(self.data_full.x)
            num_nodes = node_end - node_start

            g = Data(
                x=self.data_full.x[node_start:node_end],
                edge_index=self.data_full.edge_index[:, node_start:node_end] if self.data_full.edge_index.shape[1] >= node_end else self.data_full.edge_index,
                edge_attr=self.data_full.edge_attr[node_start:node_end] if self.data_full.edge_attr is not None and self.data_full.edge_attr.shape[0] >= node_end else None,
                y=torch.tensor(label, dtype=torch.long),
            )
            return g

    dataset = JSLibsDataset(data_full, data_dict, train_indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch_data in loader:
            batch = Batch.from_data_list(batch_data).to(device)
            out = model(batch)

            if isinstance(out, tuple):
                embeddings = out[0]
            else:
                embeddings = out

            embeddings = embeddings.cpu()
            labels_batch = batch.y.cpu().numpy()

            for emb, lbl in zip(embeddings, labels_batch):
                lbl_int = int(lbl)
                if 0 <= lbl_int < num_classes:
                    support_embeddings[lbl_int].append(emb)

            log.info("Processed batch: %d samples", len(labels_batch))

    prototypes = torch.zeros(num_classes, embedding_dim, dtype=torch.float32)
    valid_classes = 0

    for c in range(num_classes):
        if support_embeddings[c]:
            stacked = torch.stack(support_embeddings[c], dim=0)
            prototype = stacked.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes[c] = prototype
            valid_classes += 1
            log.info("Class %d: %d support embeddings, prototype norm=%.4f",
                     c, len(support_embeddings[c]), prototype.norm().item())
        else:
            log.warning("No support embeddings for class %d", c)

    log.info("Computed %d valid prototypes (out of %d classes)", valid_classes, num_classes)

    torch.save({
        'prototypes': prototypes,
        'label_map': label_map,
    }, output_path)
    log.info("Prototypes saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    main()