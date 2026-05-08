"""
graphgps/loader/dataset/cpg_vocab.py

Builds a CPG node-label vocabulary from the raw dataset and saves it
alongside split.json.  The CPGNodeEncoder then uses nn.Embedding instead
of a hash projection, eliminating collision-based feature loss.

Usage
-----
# Normal run
python -m graphgps.loader.dataset.cpg_vocab --raw_dir datasets/JSLibs/raw

# Debug: inspect what keys/attrs are actually in your XML files
python -m graphgps.loader.dataset.cpg_vocab --raw_dir datasets/JSLibs/raw --inspect

# If your XML uses a non-standard label key (e.g. CODE, NAME):
python -m graphgps.loader.dataset.cpg_vocab --raw_dir datasets/JSLibs/raw --label_key CODE

# Build vocab from custom data_dir, specific libs + bundlers
python -m graphgps.loader.dataset.cpg_vocab \
    --raw_dir  datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --lib   async axios lodash express chalk commander react request rxjs uuid  \
    --bundler rollup@4.46.2 webpack@5.95.0 \
    --verbose

# Inspect file structure to find the right --label_key
# --max_bundles bundles is more than enough to saturate CPG vocab
python -m graphgps.loader.dataset.cpg_vocab \
    --raw_dir  datasets/JSLibs/raw \
    --data_dir /home/aiuser4/ado/bundled-js-scan/data/train/v2.2 \
    --lib   async axios lodash express chalk commander react request rxjs uuid  \
    --bundler rollup@4.46.2 webpack@5.95.0 \
    --inspect \
    --max_bundles 50

# Then remove processed vocab before run training:
rm -rf datasets/JSLibs/processed/
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter
import re


# ── filter helpers (mirrors build_split.py / eda.py) ─────────────────────────

_BUNDLER_PREFIXES = ("rollup", "webpack", "vite",
                     "parcel", "esbuild", "browserify")
_DOT_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')

def _is_lib_dir(name: str, parent: str) -> bool:
    if not osp.isdir(osp.join(parent, name)):
        return False
    low = name.lower()
    if any(low.startswith(p) for p in _BUNDLER_PREFIXES):
        return False
    return name not in ("node_modules", ".git", "__pycache__",
                        "raw", "processed")


def _lib_matches(lib_ver: str, lib_filter: list) -> bool:
    """True when lib_filter is empty or lib_ver matches (exact or base name)."""
    if not lib_filter:
        return True
    base = ("@" + lib_ver.split("@")[1]
            if lib_ver.startswith("@") else lib_ver.split("@")[0])
    return lib_ver in lib_filter or base in lib_filter


def _bundler_matches(bundler_ver: str, bundler_filter: list) -> bool:
    """True when bundler_filter is empty or bundler_ver matches (exact or base)."""
    if not bundler_filter:
        return True
    bname = bundler_ver.split("@")[0]
    return bundler_ver in bundler_filter or bname in bundler_filter
from typing import Dict, List, Optional

import pydot
import xml.etree.ElementTree as ET


# ── XML helpers ───────────────────────────────────────────────────────────────

def _graphml_key_map(root: ET.Element) -> Dict[str, str]:
    """
    GraphML files declare key ids via <key id="dX" attr.name="label" .../>
    Build a map  attr_name → key_id  and  key_id → attr_name.
    """
    ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    prefix = f"{{{ns}}}" if ns else ""
    mapping = {}
    for key_elem in root.findall(f"{prefix}key"):
        kid  = key_elem.get("id", "")
        name = key_elem.get("attr.name", "")
        if kid and name:
            mapping[name] = kid   # attr_name → key_id
            mapping[kid]  = name  # key_id    → attr_name
    return mapping


def _labels_from_xml(path: str, label_key: str = "label") -> List[str]:
    """
    Extract node label strings from a GraphML / GEXF / custom XML file.

    Strategy (in order):
    1. GraphML: resolve label_key through <key> declarations → find <data key="dX">
    2. Direct <data key="label"> (or whatever label_key is)
    3. Attribute on <node> element directly
    """
    labels = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Detect namespace
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        prefix = f"{{{ns}}}" if ns else ""

        # Build GraphML key map (empty dict if not GraphML)
        key_map  = _graphml_key_map(root)
        # Resolve the label_key to the actual key id used in <data> elements
        # e.g. label_key="label" → key_map["label"] = "d3" → look for <data key="d3">
        data_key = key_map.get(label_key, label_key)  # fallback: use as-is

        for node in root.findall(f".//{prefix}node"):
            found = False

            # 1. Look for <data key="data_key"> (GraphML style)
            for d in node.findall(f"{prefix}data"):
                k = d.get("key", "")
                if k == data_key or k == label_key:
                    if d.text and d.text.strip():
                        labels.append(d.text.strip())
                        found = True
                        break

            if found:
                continue

            # 2. Try direct attribute (GEXF / custom XML style)
            for attr_name in (label_key, "label", "LABEL", "name", "NAME",
                               "code", "CODE", "type", "TYPE"):
                val = node.get(attr_name, "").strip()
                if val:
                    labels.append(val)
                    break

    except Exception:
        pass
    return labels


# iterparse instead of ET.parse — streaming, ~10× faster for large files
def _labels_from_xml_fast(path: str, label_key: str = "label") -> List[str]:
    """
    Stream-parse XML with iterparse — never loads the full DOM.
    Resolves GraphML key aliases on first pass, then streams nodes.
    """
    labels = []
    # First pass: collect key map (only <key> elements, very fast)
    key_map = {}
    try:
        for event, elem in ET.iterparse(path, events=("start",)):
            tag = elem.tag.split("}")[-1]  # strip namespace
            if tag == "key":
                kid  = elem.get("id", "")
                name = elem.get("attr.name", "")
                if kid and name:
                    key_map[name] = kid
                    key_map[kid]  = name
            elif tag == "graph":
                break   # key declarations always come before graph body
            elem.clear()

        data_key = key_map.get(label_key, label_key)

        # Second pass: stream node elements only
        for event, elem in ET.iterparse(path, events=("end",)):
            tag = elem.tag.split("}")[-1]
            if tag == "node":
                for child in elem:
                    ctag = child.tag.split("}")[-1]
                    if ctag == "data":
                        k = child.get("key", "")
                        if k in (data_key, label_key) and child.text:
                            labels.append(child.text.strip())
                            break
                elem.clear()   # ← free memory immediately
    except Exception:
        pass
    return labels


def _labels_from_dot(path: str) -> List[str]:
    """Extract node label strings from a DOT file."""
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
            label = attrs.get("label", "").strip('"').strip()
            if label:
                labels.append(label)
    except Exception:
        pass
    return labels


def _labels_from_dot_fast(path: str) -> List[str]:
    """
    Read DOT file as plain text and regex-extract label values.
    ~100× faster than pydot for large files.
    """
    labels = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = _DOT_LABEL_RE.search(line)
                if m:
                    labels.append(m.group(1))
    except Exception:
        pass
    return labels


# ── Inspection helper ─────────────────────────────────────────────────────────

def inspect_file(path: str, max_nodes: int = 5):
    """Print raw structure of the first few nodes so you can see what keys exist."""
    print(f"\n{'─'*60}")
    print(f"File: {path}")
    print(f"{'─'*60}")

    if path.endswith(".dot"):
        try:
            graphs = pydot.graph_from_dot_file(path)
            if graphs:
                nodes = graphs[0].get_nodes()
                for node in nodes[:max_nodes]:
                    if node.get_name() in ("node", "graph", "edge"):
                        continue
                    print(f"  node: {node.get_name()!r}")
                    print(f"  attrs: {node.get_attributes()}")
        except Exception as e:
            print(f"  DOT parse error: {e}")
        return

    if path.endswith(".xml"):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            ns   = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
            prefix = f"{{{ns}}}" if ns else ""

            print(f"  Root tag: {root.tag}")
            print(f"  Namespace: {ns!r}")

            key_map = _graphml_key_map(root)
            if key_map:
                print(f"  GraphML keys: {key_map}")

            count = 0
            for node in root.findall(f".//{prefix}node"):
                if count >= max_nodes:
                    break
                print(f"\n  <node id={node.get('id')!r} attrs={dict(node.attrib)}>")
                for child in node:
                    print(f"    <{child.tag}  key={child.get('key')!r}"
                          f"  text={child.text!r}>")
                count += 1
        except Exception as e:
            print(f"  XML parse error: {e}")


# ── Main vocab builder ────────────────────────────────────────────────────────

def build_vocab(
    data_dir: str,
    min_freq: int = 1,
    label_key: str = "label",
    verbose: bool = False,
    lib_filter: list = None,
    bundler_filter: list = None,
    max_bundles: Optional[int] = None,
) -> Dict[str, int]:
    """
    Walk every graph file under data_dir and collect node label frequencies.
    Returns  {label: index}  with UNK=0.

    lib_filter    : list of lib@ver or base names to include (empty = all).
    bundler_filter: list of bundler@ver or base names to include (empty = all).
    """
    lib_filter     = lib_filter or []
    bundler_filter = bundler_filter or []
    counter: Counter = Counter()
    n_files = 0
    n_empty = 0

    for lib_ver in sorted(os.listdir(data_dir)):
        lib_dir = osp.join(data_dir, lib_ver)
        if not _is_lib_dir(lib_ver, data_dir):
            continue
        if not _lib_matches(lib_ver, lib_filter):
            continue
        for bundler_ver in sorted(os.listdir(lib_dir)):
            if not osp.isdir(osp.join(lib_dir, bundler_ver)):
                continue
            if not _bundler_matches(bundler_ver, bundler_filter):
                continue
            graphs_dir = osp.join(lib_dir, bundler_ver, "graphs")
            if not osp.isdir(graphs_dir):
                continue
            # ── KEY CHANGE 1: only read _program file ──────────────────
            prog_file = None
            for ext in (".xml", ".dot"):
                candidate = osp.join(graphs_dir, f"_program{ext}")
                if osp.isfile(candidate):
                    prog_file = candidate
                    break

            if prog_file is None:
                continue   # no whole-program CPG, skip
            #
            # ── KEY CHANGE 2: streaming XML parse (iterparse) ──────────
            if prog_file.endswith(".xml"):
                labels = _labels_from_xml_fast(prog_file, label_key)
            else:
                labels = _labels_from_dot_fast(prog_file)

            counter.update(labels)
            n_bundles += 1
            #
            # ── KEY CHANGE 3: early exit once vocab saturates ──────────
            if max_bundles and n_bundles >= max_bundles:
                print(f"[early stop] reached {max_bundles} bundles, "
                      f"{len(counter)} unique labels so far")
                break
            #
            # for fname in os.listdir(graphs_dir):
            #     fpath = osp.join(graphs_dir, fname)
            #     if fname.endswith(".dot"):
            #         labels = _labels_from_dot(fpath)
            #     elif fname.endswith(".xml"):
            #         labels = _labels_from_xml(fpath, label_key=label_key)
            #     else:
            #         continue
            #     n_files += 1
            #     if not labels:
            #         n_empty += 1
            #         if verbose:
            #             print(f"  [EMPTY] {lib_ver}/{bundler_ver}/{fname}")
            #     counter.update(labels)

    print(f"\nScanned {n_files} files  |  {n_empty} returned no labels")
    print(f"Unique labels found: {len(counter)}")
    if len(counter) == 0:
        print("\n⚠️  No labels extracted. Run with --inspect to see your file structure,")
        print("   then re-run with --label_key <correct_key_name>.")

    if verbose and counter:
        print("\nTop 20 labels by frequency:")
        for label, freq in counter.most_common(20):
            print(f"  {freq:6d}  {label!r}")

    # Build vocab: UNK=0, then sorted by freq descending
    vocab: Dict[str, int] = {"UNK": 0}
    for label, freq in counter.most_common():
        if freq >= min_freq and label not in vocab:
            vocab[label] = len(vocab)

    return vocab


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path) as f:
        return json.load(f)


def label_to_idx(label: str, vocab: Dict[str, int]) -> int:
    return vocab.get(label, 0)   # 0 = UNK


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build CPG node-label vocabulary for JSLibs dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",   default="datasets/JSLibs/raw",
                        help="Path to raw/ directory — "
                             "cpg_vocab.json is always written here")
    parser.add_argument("--data_dir",  default=None,
                        help="Directory containing lib@ver/ graph subdirs. "
                             "Defaults to --raw_dir when not set.")
    parser.add_argument("--lib",       nargs="+", default=[], metavar="LIB",
                        help="Only include these libs when building the vocab. "
                             "Accepts exact (axios@1.7.9) or base (axios) names.")
    parser.add_argument("--bundler",   nargs="+", default=[], metavar="BUNDLER",
                        help="Only include these bundlers when building the vocab. "
                             "Accepts exact (rollup@4.46.2) or base (rollup) names.")
    parser.add_argument("--out",       default=None,
                        help="Output path (default: <raw_dir>/cpg_vocab.json)")
    parser.add_argument("--min_freq",  type=int, default=1,
                        help="Minimum occurrences for a label to enter the vocab")
    parser.add_argument("--label_key", default="label",
                        help="XML attribute/data key name that holds the node label. "
                             "Common values: label, LABEL, code, CODE, name, NAME")
    parser.add_argument("--inspect",   action="store_true",
                        help="Print raw structure of first few graph files and exit. "
                             "Use this to find the correct --label_key.")
    parser.add_argument("--max_bundles", type=int, default=None,
                        help="For quick iteration: max number of bundles to scan "
                             "before early exit (default: scan all).")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print files that returned no labels + top-20 label list")
    args = parser.parse_args()

    # ── resolve paths ──
    data_dir       = args.data_dir or args.raw_dir
    lib_filter     = args.lib     or []
    bundler_filter = args.bundler or []

    print(f"raw_dir   : {osp.abspath(args.raw_dir)}")
    print(f"data_dir  : {osp.abspath(data_dir)}")
    print(f"libs      : {lib_filter     if lib_filter     else 'ALL'}")
    print(f"bundlers  : {bundler_filter if bundler_filter else 'ALL'}")

    # ── inspect mode: show raw file structure ──
    if args.inspect:
        print("\nInspecting first graph file found per lib...")
        count = 0
        for lib_ver in sorted(os.listdir(data_dir)):
            lib_dir = osp.join(data_dir, lib_ver)
            if not _is_lib_dir(lib_ver, data_dir):
                continue
            if not _lib_matches(lib_ver, lib_filter):
                continue
            for bundler_ver in sorted(os.listdir(lib_dir)):
                if not osp.isdir(osp.join(lib_dir, bundler_ver)):
                    continue
                if not _bundler_matches(bundler_ver, bundler_filter):
                    continue
                graphs_dir = osp.join(lib_dir, bundler_ver, "graphs")
                if not osp.isdir(graphs_dir):
                    continue
                for fname in sorted(os.listdir(graphs_dir)):
                    if fname.endswith((".dot", ".xml")) and "Zone" not in fname:
                        inspect_file(osp.join(graphs_dir, fname))
                        count += 1
                        break
                if count:
                    break
            if count >= 3:
                break
        print("\nRe-run with --label_key <key_name_shown_above> to build the vocab.")
        return

    # ── normal vocab build ──
    vocab    = build_vocab(
        data_dir       = data_dir,
        min_freq       = args.min_freq,
        label_key      = args.label_key,
        verbose        = args.verbose,
        lib_filter     = lib_filter,
        bundler_filter = bundler_filter,
        max_bundles    = args.max_bundles,
    )
    # vocab is always written to raw_dir so jslibs.py can find it
    out_path = args.out or osp.join(args.raw_dir, "cpg_vocab.json")
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=False)
    print(f"\nWrote vocab ({len(vocab)} entries) → {out_path}")


if __name__ == "__main__":
    main()