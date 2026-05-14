from graphgps.inference.prototype_inference import (
    PrototypeInference,
    compute_prototypes_from_data,
)
from graphgps.network.gps_model import GPSModel

if __name__ == "__main__":
    import torch
    from yacs.config import CfgNode
    from torch_geometric.graphgym.config import set_cfg

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prototype_path = "results/jslibs-10libs-hash/prototypes.pt"
    model_path = "results/jslibs-10libs-hash/0/ckpt/192.ckpt"

    # Step 1: Set up cfg for model instantiation (required by GPSModel)
    _cfg = CfgNode()
    set_cfg(_cfg)
    _cfg.gnn.dim_inner = 128
    _cfg.gt.dim_hidden = 128
    _cfg.gt.layer_type = 'GatedGCN+Transformer'
    _cfg.gt.layers = 6
    _cfg.gt.n_heads = 8
    _cfg.gt.dropout = 0.0
    _cfg.gt.attn_dropout = 0.0
    _cfg.gt.layer_norm = True
    _cfg.gt.batch_norm = False
    _cfg.gt.pna_degrees = [4]
    _cfg.gt.bigbird = {}
    _cfg.gnn.layers_pre_mp = 0
    _cfg.gnn.head = 'prototype'
    _cfg.dataset.node_encoder = False
    _cfg.dataset.edge_encoder = False

    # Step 2: Instantiate model and load state_dict
    model = GPSModel(dim_in=128, dim_out=128)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    # Step 3: Compute prototypes from processed data
    compute_prototypes_from_data(
        data_path="datasets/JSLibs/processed-10libs-hash/data.pt",
        split_dict_path="datasets/JSLibs/processed-10libs-hash/split_dict.pt",
        model_path=model,  # Pass model object, not path
        output_path=prototype_path,
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
        device=device,
    )

    infer = PrototypeInference(
        model_path="dummy",
        prototypes_path=prototype_path,
        load_type="individual",
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
        device=device,
    )
    infer.set_model(model)

    results = infer.predict(
        "datasets/JSLibs/test/10libs/async-axios-chalk-debug-lodash/",
        topk=3,
        topk_per_graph=3,
    )
    infer.print_results(results)

    infer.save_results(results, "tmp/infer_results.json")