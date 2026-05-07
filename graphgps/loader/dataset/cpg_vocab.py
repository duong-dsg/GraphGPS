"""
graphgps/loader/dataset/cpg_vocab.py

Builds a CPG node-label vocabulary from the raw dataset and saves it
alongside split.json.  The CPGNodeEncoder then uses nn.Embedding instead
of a hash projection, eliminating collision-based feature loss.

Usage
-----
# Step 1: build vocab ONCE before processing the dataset
python -m graphgps.loader.dataset.cpg_vocab \
    --raw_dir datasets/JSLibs/raw \
    --out     datasets/JSLibs/raw/cpg_vocab.json

# Step 2: JSLibsDataset.process() picks it up automatically if it exists
#         at <raw_dir>/cpg_vocab.json

Vocab file format
-----------------
{
  "UNK":  0,          # always index 0
  "<METHOD>": 1,
  "CALL": 2,
  ...
}
-----------------
Build the vocab, then delete processed/ directory:
(note the vocab size printed, e.g. "Vocabulary size: 312")

python -m graphgps.loader.dataset.cpg_vocab --raw_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2
rm -rf datasets/JSLibs/processed/
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter
from typing import Dict

import pydot
import xml.etree.ElementTree as ET


# ---- label extraction ----

def _labels_from_dot(path: str) -> list:
    labels = []
    try:
        graphs = pydot.graph_from_dot_file(path)
        if not graphs:
            return labels
        for node in graphs[0].get_nodes():
            name = node.get_name()
            if name in ("node", "graph", "edge"):
                continue
            attrs = node.get_attributes()
            label = attrs.get("label", "").strip('"')
            if label:
                labels.append(label)
    except Exception:
        pass
    return labels


def _labels_from_xml(path: str) -> list:
    labels = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for node in root.findall(".//node"):
            # GraphML / GEXF: label in <data key="label">
            for d in node.findall(".//data"):
                if d.get("key") == "label" and d.text:
                    labels.append(d.text.strip())
                    break
            # fallback: label attribute
            else:
                label = node.get("label", "").strip()
                if label:
                    labels.append(label)
    except Exception:
        pass
    return labels


def build_vocab(raw_dir: str, min_freq: int = 1) -> Dict[str, int]:
    """
    Walk every graph file under raw_dir and collect node label frequencies.
    Returns  {label: index}  with UNK=0.
    """
    counter: Counter = Counter()

    for lib_ver in sorted(os.listdir(raw_dir)):
        lib_dir = osp.join(raw_dir, lib_ver)
        if not osp.isdir(lib_dir):
            continue
        for bundler_ver in sorted(os.listdir(lib_dir)):
            graphs_dir = osp.join(lib_dir, bundler_ver, "graphs")
            if not osp.isdir(graphs_dir):
                continue
            for fname in os.listdir(graphs_dir):
                fpath = osp.join(graphs_dir, fname)
                if fname.endswith(".dot"):
                    labels = _labels_from_dot(fpath)
                elif fname.endswith(".xml"):
                    labels = _labels_from_xml(fpath)
                else:
                    continue
                counter.update(labels)

    # Build index: UNK=0, then sorted by freq descending for stable ordering
    vocab: Dict[str, int] = {"UNK": 0}
    for label, freq in counter.most_common():
        if freq >= min_freq and label not in vocab:
            vocab[label] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}  "
          f"(from {sum(counter.values())} node labels, "
          f"min_freq={min_freq})")
    return vocab


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path) as f:
        return json.load(f)


def label_to_idx(label: str, vocab: Dict[str, int]) -> int:
    return vocab.get(label, 0)   # 0 = UNK


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Build CPG node-label vocabulary")
    parser.add_argument("--raw_dir", default="datasets/JSLibs/raw")
    parser.add_argument("--out",     default=None,
                        help="Output path (default: <raw_dir>/cpg_vocab.json)")
    parser.add_argument("--min_freq", type=int, default=1,
                        help="Min occurrences for a label to enter vocab")
    args = parser.parse_args()

    vocab    = build_vocab(args.raw_dir, min_freq=args.min_freq)
    out_path = args.out or osp.join(args.raw_dir, "cpg_vocab.json")
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=False)
    print(f"Wrote vocab ({len(vocab)} entries) → {out_path}")


if __name__ == "__main__":
    main()