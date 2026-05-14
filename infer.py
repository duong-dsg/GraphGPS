"""
infer.py — JSLibs Prototype-Based Inference

Usage:
    python infer.py --cfg configs/custom/jslibs-inference.yaml

Loads config from YAML, builds model, runs inference on test data.
"""

import argparse
import logging
import os.path as osp
from pathlib import Path

import torch

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, dump_cfg, set_cfg, load_cfg

import graphgps  # noqa, register custom modules
from graphgps.inference.prototype_inference import (
    PrototypeInference,
    compute_prototypes_from_data,
)
from graphgps.network.gps_model import GPSModel


def main():
    # ---- Parse cmd line args (use PyG parse_args for --cfg support) ----
    args = parse_args()

    # Add custom args
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute_prototypes", action="store_true",
                        help="Recompute prototypes from training data")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory")
    extra_args = parser.parse_known_args()[0]

    # ---- Load config from YAML ----
    set_cfg(cfg)
    load_cfg(cfg, args)
    dump_cfg(cfg)

    device = cfg.inference.device if hasattr(cfg, "inference") and hasattr(cfg.inference, "device") else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    # ---- Build model and load checkpoint ----
    logging.info("Building model...")
    model = GPSModel(dim_in=cfg.gnn.dim_inner, dim_out=cfg.gnn.dim_inner)

    ckpt_path = cfg.inference.model_ckpt
    logging.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # ---- Compute prototypes (optional) ----
    prototypes_path = cfg.inference.prototypes
    if extra_args.compute_prototypes or not osp.exists(prototypes_path):
        logging.info("Computing prototypes from training data...")
        compute_prototypes_from_data(
            data_path=cfg.inference.data_path,
            split_dict_path=cfg.inference.split_dict_path,
            model_path=model,
            output_path=prototypes_path,
            split_json=cfg.inference.prototypes_split_json,
            device=device,
        )
        logging.info(f"Prototypes saved to {prototypes_path}")

    # ---- Run inference ----
    logging.info("Setting up inference pipeline...")
    infer = PrototypeInference(
        model_path="dummy",
        prototypes_path=prototypes_path,
        load_type=cfg.inference.load_type,
        split_json=cfg.inference.prototypes_split_json,
        device=device,
    )
    infer.set_model(model)

    input_path = cfg.inference.input_path
    topk = cfg.inference.topk if hasattr(cfg.inference, "topk") else 5
    topk_per_graph = cfg.inference.topk_per_graph if hasattr(cfg.inference, "topk_per_graph") else 3

    logging.info(f"Running inference on: {input_path}")
    results = infer.predict(input_path, topk=topk, topk_per_graph=topk_per_graph)
    infer.print_results(results)

    output_json = cfg.inference.output_json
    if extra_args.out_dir:
        output_json = osp.join(extra_args.out_dir, osp.basename(output_json))
    infer.save_results(results, output_json)
    logging.info(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()