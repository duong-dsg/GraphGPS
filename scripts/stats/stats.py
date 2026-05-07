"""
stats.py — Visualise GraphGPS training results

Reads the output directory produced by GraphGPS custom_train and plots:
  1. Loss curves      (train / val / test per epoch)
  2. Accuracy curves  (accuracy + accuracy-JS if present)
  3. F1 / AUC curves
  4. LR schedule
  5. Best epoch summary bar chart
  6. Per-run comparison (multiple versions)

Directory layout expected
-------------------------
results/
  libs_scan-v2/
    0/
      agg/
        train/  stats.json  best.json
        val/    stats.json  best.json
        test/   stats.json  best.json
      train/    stats.json  best.json   (raw, optional)
      val/      ...
      test/     ...
      logging.log
    config.yaml

Usage
-----
# Single run
python scripts/visualize/stats.py --run_dir results/libs_scan-v2

# Save figures
python scripts/visualize/stats.py --run_dir results/jslibs-v4 --save_dir scripts/plots/

# Compare multiple versions
python scripts/visualize/stats.py --run_dir results \
    --compare libs_scan-v1 libs_scan-v2 libs_scan-v3
"""

import argparse
import json
import os
import os.path as osp
import sys
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    sys.exit("matplotlib required:  pip install matplotlib")


# =============================================================================
# Style constants
# =============================================================================

SPLIT_COLOR = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLIT_LS    = {"train": "-",        "val": "--",       "test": ":"}
MARKER      = {"train": "o",        "val": "s",        "test": "^"}

METRIC_LABEL = {
    "accuracy"   : "Accuracy",
    "accuracy-JS": "Accuracy-JS (balanced)",
    "f1"         : "F1 macro",
    "f1_macro"   : "F1 macro",
    "f1_weighted": "F1 weighted",
    "auc"        : "AUC",
    "ap"         : "Average Precision",
    "loss"       : "Loss",
    "lr"         : "Learning Rate",
}

SEP  = "─" * 65
SEP2 = "═" * 65

# metrics shown in summary / comparison (in this order)
KEY_METRICS = ["accuracy", "accuracy-JS", "f1", "f1_macro",
               "f1_weighted", "auc", "ap"]

# groups of metrics shown in the same subplot panel
PANEL_GROUPS = [
    ("Loss",         ["loss"]),
    ("Accuracy",     ["accuracy", "accuracy-JS"]),
    ("F1 / AUC",     ["f1", "f1_macro", "f1_weighted", "auc", "ap"]),
    ("LR schedule",  ["lr"]),
]


# =============================================================================
# I/O
# =============================================================================

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_stats(path: str) -> List[dict]:
    """Load stats.json — supports JSON array or JSONL."""
    if not osp.exists(path):
        return []
    try:
        with open(path) as f:
            content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
            # each element may itself be a dict or a list of dicts
            flat = []
            for item in data:
                if isinstance(item, dict):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(item)
            return flat
        else:
            rows = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
    except Exception as e:
        print(f"  [WARN] Could not parse {path}: {e}")
        return []


def find_repeat_dirs(run_dir: str) -> List[str]:
    """Return list of repeat subdirectories (0/, 1/, …) or [run_dir] itself."""
    if osp.isdir(osp.join(run_dir, "agg")):
        return [run_dir]
    candidates = sorted(
        d for d in os.listdir(run_dir)
        if d.isdigit() and osp.isdir(osp.join(run_dir, d))
    )
    return [osp.join(run_dir, d) for d in candidates] if candidates else [run_dir]


def load_run(repeat_dir: str) -> Dict[str, Dict]:
    """
    Load {split: {stats: [...], best: {...}}} for one repeat directory.
    Prefers agg/<split>/ over raw <split>/ if both exist.
    """
    run = {}
    for split in ("train", "val", "test"):
        for subdir in (osp.join(repeat_dir, "agg", split),
                       osp.join(repeat_dir, split)):
            if not osp.isdir(subdir):
                continue
            stats = _load_stats(osp.join(subdir, "stats.json"))
            best  = _load_json(osp.join(subdir, "best.json")) or {}
            if stats or best:
                run[split] = {"stats": stats, "best": best}
                break
    return run


def metric_series(stats: List[dict], key: str) -> Tuple[List, List]:
    """Extract (epochs, values) for a metric, skipping missing entries."""
    epochs, vals = [], []
    for d in stats:
        if key in d and d[key] is not None:
            epochs.append(d.get("epoch", len(epochs)))
            vals.append(float(d[key]))
    return epochs, vals


def all_metric_keys(run: Dict) -> List[str]:
    """All numeric metric keys present in any split, ordered sensibly."""
    found = set()
    skip  = {"epoch", "time_epoch", "time_iter", "params",
              "eta", "eta_hours", "gpu_memory"}
    for split_data in run.values():
        for row in split_data.get("stats", []):
            for k, v in row.items():
                if k not in skip and isinstance(v, (int, float)):
                    found.add(k)
    priority = ["loss", "accuracy", "accuracy-JS",
                "f1", "f1_macro", "f1_weighted", "auc", "ap", "lr"]
    ordered  = [k for k in priority if k in found]
    ordered += sorted(k for k in found if k not in priority)
    return ordered


# =============================================================================
# Helpers
# =============================================================================

def _style(ax, title, ylabel, xlabel="Epoch"):
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(fontsize=7.5, framealpha=0.85)
    if leg:
        leg.get_frame().set_linewidth(0.4)


def _save_or_show(fig, save_dir: str, fname: str):
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = osp.join(save_dir, fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  saved → {out}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Plot 1 — Training curves
# =============================================================================

def plot_curves(run: Dict, run_name: str, save_dir: str):
    avail = all_metric_keys(run)
    if not avail:
        print("  [WARN] No metrics found."); return

    best_epoch = (run.get("val", {}).get("best") or {}).get("epoch")

    # build panels — only include groups that have at least one available metric
    panels = [(title, [m for m in mkeys if m in avail])
              for title, mkeys in PANEL_GROUPS]
    panels = [(t, ms) for t, ms in panels if ms]
    # catch any metrics not in predefined groups
    covered = {m for _, ms in panels for m in ms}
    extra   = [m for m in avail if m not in covered]
    if extra:
        panels.append(("Other", extra))

    n = len(panels)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    axes = [axes] if n == 1 else list(axes)

    line_styles = ["-", "--", "-.", ":"]

    for ax, (panel_title, mkeys) in zip(axes, panels):
        for split in ("train", "val", "test"):
            if split not in run:
                continue
            for idx, metric in enumerate(mkeys):
                ep, vals = metric_series(run[split]["stats"], metric)
                if not vals:
                    continue
                ls    = SPLIT_LS[split]
                # second+ metric in same panel gets different line style
                if len(mkeys) > 1:
                    ls = line_styles[idx % len(line_styles)]
                label = f"{split} {METRIC_LABEL.get(metric, metric)}"
                ax.plot(ep, vals,
                        color=SPLIT_COLOR[split],
                        linestyle=ls, linewidth=1.4,
                        marker=MARKER[split], markersize=2.5,
                        markevery=max(1, max((len(ep)//20), 1)),
                        label=label, alpha=0.9)

        if best_epoch is not None:
            ax.axvline(best_epoch, color="#333", linestyle="--",
                       linewidth=0.9, alpha=0.55,
                       label=f"best ep={best_epoch}")

        ylabel = (METRIC_LABEL.get(mkeys[0], mkeys[0])
                  if len(mkeys) == 1 else panel_title)
        _style(ax, panel_title, ylabel)

    fig.suptitle(f"Training Curves — {run_name}",
                 fontsize=12, fontweight="bold", y=1.01)
    _save_or_show(fig, save_dir, "01_curves.png")


# =============================================================================
# Plot 2 — Best-epoch bar chart
# =============================================================================

def plot_best_bar(run: Dict, run_name: str, save_dir: str):
    # collect per-split best values
    data = {}
    for split in ("train", "val", "test"):
        best = (run.get(split) or {}).get("best", {})
        row  = {m: best[m] for m in KEY_METRICS if m in best}
        if row:
            data[split] = row

    if not data:
        return

    present = [m for m in KEY_METRICS
               if any(m in v for v in data.values())]
    if not present:
        return

    splits  = [s for s in ("train", "val", "test") if s in data]
    x       = np.arange(len(present))
    w       = 0.22
    offsets = np.linspace(-(len(splits)-1)*w/2,
                           (len(splits)-1)*w/2, len(splits))

    fig, ax = plt.subplots(figsize=(max(7, len(present)*2), 5))

    for sp, off in zip(splits, offsets):
        vals = [data[sp].get(m, 0.0) for m in present]
        bars = ax.bar(x + off, vals, w,
                      label=sp, color=SPLIT_COLOR[sp],
                      alpha=0.82, edgecolor="white", linewidth=0.7)
        for bar, val in zip(bars, vals):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL.get(m, m) for m in present],
                        rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, min(1.22, ax.get_ylim()[1] * 1.22))
    ax.set_title(f"Best-Epoch Metrics — {run_name}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, save_dir, "02_best_bar.png")


# =============================================================================
# Plot 3 — Timing
# =============================================================================

def plot_timing(run: Dict, run_name: str, save_dir: str):
    if "train" not in run:
        return
    ep, times = metric_series(run["train"]["stats"], "time_epoch")
    if not times:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    ax1.plot(ep, times, color=SPLIT_COLOR["train"],
             linewidth=1.2, marker="o", markersize=2.5)
    ax1.set_title("Train time per epoch  (s)", fontsize=10)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("seconds")
    ax1.grid(True, alpha=0.22)
    ax1.spines[["top","right"]].set_visible(False)

    cum = np.cumsum(times)
    ax2.fill_between(ep, cum / 60, alpha=0.35, color="#8172B2")
    ax2.plot(ep, cum / 60, color="#8172B2", linewidth=1.2)
    ax2.set_title("Cumulative time  (min)", fontsize=10)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("minutes")
    ax2.grid(True, alpha=0.22)
    ax2.spines[["top","right"]].set_visible(False)

    # annotate total
    ax2.annotate(f"total: {cum[-1]/60:.1f} min",
                 xy=(ep[-1], cum[-1]/60),
                 xytext=(-40, -15), textcoords="offset points",
                 fontsize=8, color="#8172B2",
                 arrowprops=dict(arrowstyle="->", color="#8172B2", lw=0.8))

    fig.suptitle(f"Training Time — {run_name}",
                 fontsize=11, fontweight="bold")
    _save_or_show(fig, save_dir, "03_timing.png")


# =============================================================================
# Plot 4 — Multi-run comparison
# =============================================================================

def plot_comparison(runs: Dict[str, Dict], cmp_metrics: List[str],
                    split: str, save_dir: str):
    present = [m for m in cmp_metrics
               if any(metric_series(rd.get(split,{}).get("stats",[]), m)[1]
                      for rd in runs.values())]
    if not present:
        print(f"  [WARN] No data for comparison on {split} split.")
        return

    n   = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    axes = [axes] if n == 1 else list(axes)

    palette = plt.cm.tab10.colors

    for ax, metric in zip(axes, present):
        for i, (name, rd) in enumerate(runs.items()):
            ep, vals = metric_series(rd.get(split,{}).get("stats",[]), metric)
            if not vals:
                continue
            ax.plot(ep, vals,
                    color=palette[i % len(palette)],
                    linewidth=1.4, label=name, alpha=0.9)
        _style(ax,
               f"{METRIC_LABEL.get(metric, metric)}  [{split}]",
               METRIC_LABEL.get(metric, metric))

    fig.suptitle(f"Run Comparison — {split} split",
                 fontsize=12, fontweight="bold", y=1.01)
    _save_or_show(fig, save_dir, f"04_comparison_{split}.png")


# =============================================================================
# Console summary
# =============================================================================

def print_best(run: Dict, run_name: str):
    print(f"\n{SEP2}")
    print(f"  {run_name}")
    print(SEP2)

    row_keys = ["loss"] + KEY_METRICS + ["lr"]

    for split in ("train", "val", "test"):
        best = (run.get(split) or {}).get("best", {})
        if not best:
            continue
        ep = best.get("epoch", "?")
        print(f"\n  [{split.upper()}]  best epoch = {ep}")
        print(f"  {SEP[:52]}")
        for k in row_keys:
            if k not in best:
                continue
            label = METRIC_LABEL.get(k, k)
            val   = best[k]
            bar   = ""
            if isinstance(val, float) and 0.0 <= val <= 1.0 and k != "lr":
                n = int(val * 28)
                bar = "  " + "█" * n + "░" * (28 - n)
            print(f"  {label:<24} {val:.6f}{bar}")


def print_comparison_table(runs: Dict[str, Dict]):
    print(f"\n{SEP2}")
    print("  COMPARISON TABLE  (best-epoch values)")
    print(SEP2)

    for split in ("val", "test"):
        print(f"\n  {split.upper()} SPLIT")
        col_w = 16
        header = f"  {'run':<30}" + "".join(
            f"{METRIC_LABEL.get(m, m)[:col_w-1]:<{col_w}}"
            for m in KEY_METRICS)
        print(header)
        print(f"  {SEP}")
        for name, rd in runs.items():
            best = (rd.get(split) or {}).get("best", {})
            row  = f"  {name:<30}"
            for m in KEY_METRICS:
                v = best.get(m)
                row += f"{v:.4f}          "[:col_w] if v is not None \
                       else f"{'—':<{col_w}}"
            print(row)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Visualise GraphGPS training results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",    default="results/libs_scan-v2",
                   help="Run directory (contains 0/ repeat dirs or agg/ directly)")
    p.add_argument("--compare",    nargs="+", default=[], metavar="NAME",
                   help="Sub-dir names inside --run_dir to compare side-by-side")
    p.add_argument("--split",      default="val",
                   choices=["train","val","test"],
                   help="Split used for comparison curves")
    p.add_argument("--metrics",    nargs="+", default=[], metavar="METRIC",
                   help="Metrics for comparison plot (default: accuracy, auc, f1)")
    p.add_argument("--save_dir",   default="",
                   help="Save figures here as PNG (empty = show interactively)")
    p.add_argument("--no_timing",  action="store_true",
                   help="Skip the timing plot")
    args = p.parse_args()

    if args.save_dir:
        matplotlib.use("Agg")

    # ── single-run mode ───────────────────────────────────────────────────────
    if not args.compare:
        if not osp.isdir(args.run_dir):
            sys.exit(f"[ERROR] Not found: {args.run_dir}")

        repeats = find_repeat_dirs(args.run_dir)
        print(f"\n{SEP2}")
        print(f"  Results : {osp.abspath(args.run_dir)}")
        print(f"  Repeats : {len(repeats)}")
        print(SEP2)

        for rep_dir in repeats:
            run_name = (osp.basename(args.run_dir) + "/" +
                        osp.basename(rep_dir)
                        if rep_dir != args.run_dir
                        else osp.basename(args.run_dir))
            run = load_run(rep_dir)
            if not run:
                print(f"  [WARN] No data in {rep_dir}"); continue

            print_best(run, run_name)

            rep_save = (osp.join(args.save_dir, osp.basename(rep_dir))
                        if args.save_dir and len(repeats) > 1
                        else args.save_dir)
            plot_curves(run, run_name, rep_save)
            plot_best_bar(run, run_name, rep_save)
            if not args.no_timing:
                plot_timing(run, run_name, rep_save)

    # ── comparison mode ───────────────────────────────────────────────────────
    else:
        runs = {}
        for name in args.compare:
            rdir = osp.join(args.run_dir, name)
            if not osp.isdir(rdir):
                print(f"  [WARN] Not found: {rdir}"); continue
            rep    = find_repeat_dirs(rdir)[0]
            run    = load_run(rep)
            if run:
                runs[name] = run
            else:
                print(f"  [WARN] No data in {rdir}")

        if not runs:
            sys.exit("[ERROR] No valid runs found.")

        print_comparison_table(runs)

        cmp_m = args.metrics or [
            m for m in ("accuracy", "accuracy-JS", "f1", "auc")
            if any(metric_series(rd.get(args.split,{})
                                   .get("stats",[]), m)[1]
                   for rd in runs.values())
        ] or ["accuracy", "loss"]

        for sp in ("val", "test"):
            plot_comparison(runs, cmp_m, sp, args.save_dir)

    print(f"\n{SEP2}")
    print("  Done.")
    if args.save_dir:
        print(f"  Figures: {osp.abspath(args.save_dir)}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()