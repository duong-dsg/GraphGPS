from graphgps.inference.prototype_inference import (
    PrototypeInference,
    compute_prototypes_from_data,
)
from graphgps.network.gps_model import GPSModel

if __name__ == "__main__":
    import torch
    from torch_geometric.graphgym.config import cfg_from_dict

    prototype_path = "results/jslibs-10libs-hash/prototypes.pt"
    model_path = "results/jslibs-10libs-hash/0/ckpt/192.ckpt"

    # Step 1: Compute prototypes from processed data
    # Note: compute_prototypes_from_data handles state_dict checkpoints correctly
    compute_prototypes_from_data(
        data_path="datasets/JSLibs/processed-10libs-hash/data.pt",
        split_dict_path="datasets/JSLibs/processed-10libs-hash/split_dict.pt",
        model_path=model_path,  # Pass checkpoint path - handles state_dict automatically
        output_path=prototype_path,
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
    )

    # Step 2: For inference, we need a full model object
    # Instantiate model and load state_dict from checkpoint
    cfg_from_dict({
        'gnn.dim_inner': 128,
        'gt.dim_hidden': 128,
        'gt.layer_type': 'GatedGCN+Transformer',
        'gt.layers': 6,
        'gt.n_heads': 8,
        'gt.dropout': 0.0,
        'gt.attn_dropout': 0.0,
        'gt.layer_norm': True,
        'gt.batch_norm': False,
        'gt.pna_degrees': [4],
        'gt.bigbird': {},
        'gnn.layers_pre_mp': 0,
        'gnn.head': 'prototype',
        'dataset.node_encoder': False,
        'dataset.edge_encoder': False,
    })

    model = GPSModel(dim_in=128, dim_out=128)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    # Step 3: Create inference object and set model
    infer = PrototypeInference(
        model_path="dummy",  # placeholder since we'll set_model below
        prototypes_path=prototype_path,
        load_type="individual",
        split_json="datasets/JSLibs/processed-10libs-hash/split.json",
    )
    infer.set_model(model)

    # Step 4: Run inference
    results = infer.predict(
        "datasets/JSLibs/test/10libs/async-axios-chalk-debug-lodash/",
        topk=3,
        topk_per_graph=3,
    )
    infer.print_results(results)

    infer.save_results(results, "tmp/infer_results.json")