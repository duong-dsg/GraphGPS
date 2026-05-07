from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    # ── JSLibs dataset ────────────────────────────────────────────────────────
 
    # Path to the directory containing lib@ver/ graph subdirectories.
    # Empty string → falls back to <dataset_dir>/raw/ (original behaviour).
    cfg.dataset.data_dir = ""
 
    # Path to split.json — used by the logger to show lib names instead of
    # class indices in the per-class accuracy breakdown.
    cfg.dataset.label_map_path = ""
 
    # Vocabulary size for CPGNodeEncoder embedding mode.
    # 0 → use hash/linear projection fallback.
    # Set to the value printed by cpg_vocab.py after building the vocab.
    cfg.dataset.node_encoder_vocab_size = 0
 
    # Graph size filters applied during JSLibsDataset.process().
    cfg.dataset.min_nodes = 5
    cfg.dataset.max_nodes = 2000
