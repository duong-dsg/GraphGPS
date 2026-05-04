"""
JSLibCPG Dataset
================
Custom InMemoryDataset cho JS library CPG graphs.

Cấu trúc thư mục input (data/jslib/raw/):
    data/jslib/
    ├── raw/
    │   ├── axios/
    │   │   ├── rollup@4.46.2/
    │   │   │   ├── original/
    │   │   │   │   └── graphs/          ← PyG .pt files từ Joern export
    │   │   │   │       ├── func_0.pt
    │   │   │   │       ├── func_1.pt
    │   │   │   │       └── ...
    │   │   │   └── obf_rename_mangled/
    │   │   │       └── graphs/
    │   │   └── webpack@5.95.0/
    │   │       └── ...
    │   └── lodash/
    │       └── ...
    └── split.json                       ← output của obfuscate_dataset.py

Mỗi PyG Data object cần có:
    data.x          : [N, node_feat_dim]  node features
    data.edge_index : [2, E]
    data.edge_attr  : [E, 1]              edge type (0=AST,1=CFG,2=PDG,3=CDG)
    data.y          : [1]                 lib index (dùng cho split, không phải label)
    data.lib        : str                 tên library
    data.num_nodes  : int
"""

import json
import logging
import os
import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset


log = logging.getLogger(__name__)


class JSLibDataset(InMemoryDataset):
    """
    PyG InMemoryDataset cho JS Library CPG graphs.

    Args:
        root           : thư mục gốc (vd: 'data/jslib')
                         raw/      ← input graphs từ Joern
                         processed/← cache sau khi process()
        split_path     : path đến split.json (default: root/raw/split.json)
        min_nodes      : bỏ qua subgraph có ít hơn N nodes (default: 5)
        max_nodes      : bỏ qua subgraph có nhiều hơn N nodes (default: 1000)
        transform      : PyG transform áp dụng khi __getitem__
        pre_transform  : PyG transform áp dụng trước khi lưu cache
        pre_filter     : filter trước khi lưu cache
    """

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
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # Chỉ cần split.json tồn tại để trigger process()
        return ['split.json']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_dict.pt']

    def download(self):
        # Không download — data được tạo bởi obfuscate_dataset.py + Joern
        pass

    def process(self):
        # Load lib → split mapping
        with open(self.split_path) as f:
            lib_split: dict = json.load(f)  # {lib_name: "train"|"val"|"test"}

        # Build lib → integer index (dùng làm y label cho GraphGym)
        all_libs = sorted(lib_split.keys())
        lib_to_idx = {lib: i for i, lib in enumerate(all_libs)}

        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}
        skipped = 0

        # Walk raw directory: raw/<lib>/<bundler>/<obf_id>/graphs/*.pt
        for lib in sorted(os.listdir(self.raw_dir)):
            lib_dir = osp.join(self.raw_dir, lib)
            if not osp.isdir(lib_dir) or lib not in lib_split:
                continue

            split = lib_split[lib]
            lib_idx = lib_to_idx[lib]
            split_key = 'valid' if split == 'val' else split  # PyG convention

            for bundler in sorted(os.listdir(lib_dir)):
                bundler_dir = osp.join(lib_dir, bundler)
                if not osp.isdir(bundler_dir):
                    continue

                for obf_id in sorted(os.listdir(bundler_dir)):
                    graphs_dir = osp.join(bundler_dir, obf_id, 'graphs')
                    if not osp.isdir(graphs_dir):
                        continue

                    for fname in sorted(os.listdir(graphs_dir)):
                        if not fname.endswith('.pt'):
                            continue

                        fpath = osp.join(graphs_dir, fname)
                        try:
                            g: Data = torch.load(fpath, weights_only=False)
                        except Exception as e:
                            log.warning("Failed to load %s: %s", fpath, e)
                            skipped += 1
                            continue

                        # Validate và filter
                        if g.num_nodes < self.min_nodes:
                            skipped += 1
                            continue
                        if g.num_nodes > self.max_nodes:
                            skipped += 1
                            continue
                        if g.edge_index is None or g.edge_index.shape[1] == 0:
                            skipped += 1
                            continue

                        # Gán metadata
                        g.y = torch.tensor([lib_idx], dtype=torch.long)
                        g.lib = lib
                        g.bundler = bundler
                        g.obf_id = obf_id

                        # Đảm bảo edge_attr tồn tại (fallback: AST=0)
                        if g.edge_attr is None:
                            g.edge_attr = torch.zeros(
                                g.edge_index.shape[1], 1, dtype=torch.long
                            )

                        # Đảm bảo x là float
                        if g.x is not None and g.x.dtype != torch.float:
                            g.x = g.x.float()

                        idx = len(data_list)
                        data_list.append(g)
                        split_dict[split_key].append(idx)

        log.info(
            "Loaded %d graphs (%d skipped). "
            "Train: %d, Val: %d, Test: %d",
            len(data_list), skipped,
            len(split_dict['train']),
            len(split_dict['valid']),
            len(split_dict['test']),
        )

        if self.pre_filter is not None:
            data_list = [g for g in data_list if self.pre_filter(g)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(g) for g in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])
        log.info("Saved processed dataset to %s", self.processed_dir)

    def get_idx_split(self) -> dict:
        """Returns train/valid/test index split dict."""
        return torch.load(self.processed_paths[1], weights_only=False)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"libs={len(set(self.data.lib))}, "
                f"graphs={len(self)})")