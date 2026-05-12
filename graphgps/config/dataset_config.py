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

    # Bundler filter: list of bundler@ver strings to include during
    # dataset processing. Empty list = include all bundlers.
    # Accepts exact names ('rollup@4.46.2') or base names ('rollup').
    # Example in yaml:
    #   dataset:
    #     bundler_filter: ['rollup@4.46.2', 'webpack@5.95.0']
    cfg.dataset.bundler_filter = []
 
    # Lib filter: list of lib@ver strings to include during dataset
    # processing. Empty list = include all libs.
    # Accepts exact names ('axios@1.7.9') or base names ('axios').
    # Example in yaml:
    #   dataset:
    #     lib_filter: ['axios@1.7.9', 'lodash', 'chalk@5.3.0']
    cfg.dataset.lib_filter = []

    # ── JSLibsEntire (whole-program CPG + k-hop subgraph extraction) ─────────
 
    # Which loader to use:
#   'individual' → jslibs.py    (one graph per function file)
#   'entire'     → jslibs_entire.py  (whole _program.xml + k-hop extraction)
    cfg.dataset.loader_type = "individual"
 
    # BFS hop radius for k-hop ego subgraph extraction around each
    # function-entry node. Larger = more context but bigger graphs.
    # Should roughly match gt.layers.
    cfg.dataset.max_depth = 3
 
    # Max subgraphs extracted per bundle (lib@ver + bundler@ver combo).
    # 0 = no limit. Use a small int (e.g. 20) during debugging.
    cfg.dataset.max_graphs_per_bundler = 0
 
    # Closed-mode subgraph split ratios.
    # Each _program file's subgraphs are randomly split into
    # train/val/test independently of the lib-level split.
    cfg.dataset.closed_train_ratio = 0.70
    cfg.dataset.closed_val_ratio   = 0.15
