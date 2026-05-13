"""
stats.py — Visualise GraphGPS / JSLibs training results
========================================================

Reads the output directory produced by GraphGPS custom_train and plots:
  1. Training curves  (loss / accuracy / F1+AUC / LR per epoch)
  2. Best-epoch summary bar chart
  3. Training time analysis
  4. Per-run comparison (multiple versions)

Directory layout
----------------
results/
  jslibs-v3/
    0/                       ← repeat index
      ckpt/
      train/  stats.json
      val/    stats.json
      test/   stats.json
    2/   ...
    3/   ...
    agg/                     ← aggregated across repeats
      train/  stats.json  best.json
      val/    stats.json  best.json
      test/   stats.json  best.json
    config.yaml

Sample stats.json row
---------------------
{"epoch": 20, "time_epoch": 47.58, "eta": 8584.5, "eta_hours": 2.38,
 "loss": 1.083, "lr": 0.00097, "params": 324296, "time_iter": 0.059,
 "accuracy": 0.5838, "f1": 0.41926, "accuracy-JS": 0.41047, "auc": 0.87531}

Usage
-----
# Single run — reads agg/ preferentially, falls back to first repeat
python scripts/stats/stats.py --run_dir results/jslibs-v3

# Save figures to disk
python scripts/stats/stats.py --run_dir results/jslibs-v6 --save_dir scripts/plots/v6

# Compare multiple runs
python scripts/stats/stats.py --run_dir results \\
    --compare jslibs-v2 jslibs-v3
"""

from __future__ import annotations

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
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    sys.exit("matplotlib required:  pip install matplotlib")


# =============================================================================
# Style — dark technical theme with accent colours
# =============================================================================

THEME = dict(
    bg       = "#0f1117",
    panel    = "#161b22",
    border   = "#30363d",
    text     = "#e6edf3",
    muted    = "#7d8590",
    grid     = "#21262d",
)

# Per-split palette: slightly desaturated so they read on dark bg
SPLIT_COLOR = {
    "train": "#58a6ff",   # blue
    "val"  : "#f0883e",   # amber
    "test" : "#3fb950",   # green
}
SPLIT_LS = {"train": "-", "val": "--", "test": ":"}
MARKER   = {"train": "o", "val": "s",  "test": "^"}

# Human-readable names for metric keys
METRIC_LABEL: Dict[str, str] = {
    "accuracy"   : "Accuracy",
    "accuracy-JS": "Accuracy-JS (balanced)",
    "f1"         : "F1 macro",
    "f1_macro"   : "F1 macro",
    "f1_weighted": "F1 weighted",
    "auc"        : "AUC",
    "ap"         : "Avg Precision",
    "loss"       : "Loss",
    "lr"         : "Learning Rate",
}

# Keys that are timing / admin — excluded from metric plots
SKIP_KEYS = {"epoch", "time_epoch", "time_iter", "eta", "eta_hours",
             "params", "gpu_memory"}

# Subplot panel definitions: (title, [metric_keys_in_order])
PANEL_GROUPS = [
    ("Loss",        ["loss"]),
    ("Accuracy",    ["accuracy", "accuracy-JS"]),
    ("F1 / AUC",   ["f1", "f1_macro", "f1_weighted", "auc", "ap"]),
    ("LR schedule", ["lr"]),
]

# Metrics shown in bar chart / comparison table
KEY_METRICS = ["accuracy", "accuracy-JS", "f1", "f1_macro",
               "f1_weighted", "auc", "ap"]

SEP  = "─" * 68
SEP2 = "═" * 68


# =============================================================================
# Matplotlib global style setup
# =============================================================================

def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor"      : THEME["bg"],
        "axes.facecolor"        : THEME["panel"],
        "axes.edgecolor"        : THEME["border"],
        "axes.labelcolor"       : THEME["text"],
        "axes.titlecolor"       : THEME["text"],
        "axes.grid"             : True,
        "grid.color"            : THEME["grid"],
        "grid.linewidth"        : 0.6,
        "xtick.color"           : THEME["muted"],
        "ytick.color"           : THEME["muted"],
        "text.color"            : THEME["text"],
        "legend.facecolor"      : THEME["panel"],
        "legend.edgecolor"      : THEME["border"],
        "legend.labelcolor"     : THEME["text"],
        "figure.dpi"            : 130,
        "savefig.facecolor"     : THEME["bg"],
        "savefig.bbox"          : "tight",
        "font.family"           : "monospace",
        "font.size"             : 8.5,
        "axes.spines.top"       : False,
        "axes.spines.right"     : False,
        "axes.spines.left"      : True,
        "axes.spines.bottom"    : True,
    })


# =============================================================================
# I/O helpers
# =============================================================================

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_stats(path: str) -> List[dict]:
    """
    Load stats.json — supports:
      - JSON array:  [{"epoch":1,...}, ...]
      - JSONL:       {"epoch":1,...}\n{"epoch":2,...}
      - Nested array: [[{"epoch":1,...}], ...]
    """
    if not osp.exists(path):
        return []
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            data = json.loads(content)
            flat: List[dict] = []
            for item in data:
                if isinstance(item, dict):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(x for x in item if isinstance(x, dict))
            return flat
        # JSONL
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
    """
    Return list of repeat sub-directories (0/, 2/, 3/, …).
    If none found, return [run_dir] itself.
    """
    candidates = sorted(
        d for d in os.listdir(run_dir)
        if d.isdigit() and osp.isdir(osp.join(run_dir, d))
    )
    return [osp.join(run_dir, d) for d in candidates] if candidates else [run_dir]


def load_run(run_dir: str) -> Dict[str, Dict]:
    """
    Load {split → {stats: [...], best: {...}}} for one run directory.

    Priority for stats:  agg/<split>/stats.json  >  <repeat>/<split>/stats.json
    Priority for best:   agg/<split>/best.json   (only in agg/)

    If agg/ exists use it for both. Otherwise aggregate stats from all
    repeats by averaging per-epoch values.
    """
    run: Dict[str, Dict] = {}
    agg_dir = osp.join(run_dir, "agg")

    for split in ("train", "val", "test"):
        stats: List[dict] = []
        best:  dict       = {}

        # 1. Try agg/
        if osp.isdir(agg_dir):
            agg_split = osp.join(agg_dir, split)
            if osp.isdir(agg_split):
                stats = _load_stats(osp.join(agg_split, "stats.json"))
                best  = _load_json(osp.join(agg_split, "best.json")) or {}

        # 2. Fall back to first repeat
        if not stats:
            for rep_dir in find_repeat_dirs(run_dir):
                split_dir = osp.join(rep_dir, split)
                if osp.isdir(split_dir):
                    s = _load_stats(osp.join(split_dir, "stats.json"))
                    if s:
                        stats = s
                        break

        if stats or best:
            run[split] = {"stats": stats, "best": best}

    return run


def metric_series(
    stats: List[dict], key: str
) -> Tuple[List[int], List[float]]:
    """Extract (epochs, values) for one metric key."""
    epochs, vals = [], []
    for i, d in enumerate(stats):
        if key in d and d[key] is not None:
            epochs.append(int(d.get("epoch", i)))
            vals.append(float(d[key]))
    return epochs, vals


def all_metric_keys(run: Dict) -> List[str]:
    """All numeric metric keys found across splits, in a sensible order."""
    found: set = set()
    for sd in run.values():
        for row in sd.get("stats", []):
            for k, v in row.items():
                if k not in SKIP_KEYS and isinstance(v, (int, float)):
                    found.add(k)
    priority = ["loss", "accuracy", "accuracy-JS",
                "f1", "f1_macro", "f1_weighted", "auc", "ap", "lr"]
    ordered  = [k for k in priority if k in found]
    ordered += sorted(k for k in found if k not in priority)
    return ordered


# =============================================================================
# Axis styling helper
# =============================================================================

def _style_ax(ax, title: str, ylabel: str, xlabel: str = "Epoch"):
    ax.set_title(title, fontsize=9.5, pad=7, color=THEME["text"],
                 fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=THEME["muted"], labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8, color=THEME["muted"], labelpad=4)
    ax.tick_params(axis="both", labelsize=7.5, colors=THEME["muted"],
                   length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.7)
    # Legend is placed outside by the caller via _place_legend_below().


def _place_legend_below(ax, ncol: int = 3):
    """
    Place a deduplicated legend below the axes, outside the plot area.
    Duplicate labels (smoothing ghost lines) are removed before rendering.
    """
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates while preserving insertion order
    seen: dict = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    if not seen:
        return
    leg = ax.legend(
        seen.values(), seen.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),   # just below x-axis label
        ncol=min(ncol, len(seen)),
        fontsize=7,
        framealpha=0.75,
        fancybox=False,
        edgecolor=THEME["border"],
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)


def _save_or_show(fig, save_dir: str, fname: str):
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = osp.join(save_dir, fname)
        fig.savefig(out, dpi=150)
        print(f"  saved  →  {out}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Plot 1 — Training curves
# =============================================================================

def plot_curves(run: Dict, run_name: str, save_dir: str):
    avail = all_metric_keys(run)
    if not avail:
        print("  [WARN] No metrics found — skipping curves plot.")
        return
 
    # Build panels: only groups with at least one available metric
    panels = [
        (title, [m for m in mkeys if m in avail])
        for title, mkeys in PANEL_GROUPS
    ]
    panels = [(t, ms) for t, ms in panels if ms]
    covered = {m for _, ms in panels for m in ms}
    extra   = [m for m in avail if m not in covered]
    if extra:
        panels.append(("Other", extra))
 
    n = len(panels)
    if n == 0:
        return
 
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2),
                              facecolor=THEME["bg"])
 
    if n == 1:
        axes = [axes]
 
    # Best epoch marker from val best.json
    best_ep = (run.get("val", {}).get("best") or {}).get("epoch")
 
    line_dash = ["-", "--", "-.", (0, (3,1,1,1))]
 
    for ax, (panel_title, mkeys) in zip(axes, panels):
        ax.set_facecolor(THEME["panel"])
 
        plotted = False
        for split in ("train", "val", "test"):
            if split not in run:
                continue
            stats = run[split]["stats"]
            for mi, metric in enumerate(mkeys):
                ep, vals = metric_series(stats, metric)
                if not vals:
                    continue
 
                ls = SPLIT_LS[split]
                if len(mkeys) > 1:
                    ls = line_dash[mi % len(line_dash)]
 
                label = f"{split}  {METRIC_LABEL.get(metric, metric)}"
                # smoothed background line for readability
                if len(vals) > 10:
                    smooth = np.convolve(vals, np.ones(5)/5, mode="same")
                    ax.plot(ep, smooth,
                            color=SPLIT_COLOR[split],
                            linestyle=ls, linewidth=2.0,
                            alpha=0.25, zorder=2)
                ax.plot(ep, vals,
                        color=SPLIT_COLOR[split],
                        linestyle=ls, linewidth=1.2,
                        marker=MARKER[split],
                        markersize=2.2,
                        markevery=max(1, len(ep) // 20),
                        label=label, alpha=0.92, zorder=3)
                plotted = True
 
        if not plotted:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=THEME["muted"])
 
        # Best epoch vertical line
        if best_ep is not None:
            ax.axvline(best_ep, color="#f78166", linestyle="--",
                       linewidth=1.1, alpha=0.7, zorder=1,
                       label=f"best ep={best_ep}")
 
        ylabel = (METRIC_LABEL.get(mkeys[0], mkeys[0])
                  if len(mkeys) == 1 else "Score")
        _style_ax(ax, panel_title, ylabel)
        # ncol: 3 for single-metric panels (train/val/test), more for multi
        _place_legend_below(ax, ncol=3 if len(mkeys) == 1 else 4)
 
    # Reserve space at the bottom for the below-axes legends
    fig.subplots_adjust(bottom=0.28, top=0.92, wspace=0.32)
 
    fig.suptitle(f"Training Curves  ·  {run_name}",
                 fontsize=11, fontweight="bold",
                 color=THEME["text"], x=0.01, ha="left", y=0.99)
 
    # Use savefig directly — tight_layout would fight subplots_adjust
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = osp.join(save_dir, "01_curves.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  saved  →  {out}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Plot 2 — Best-epoch bar chart
# =============================================================================

def plot_best_bar(run: Dict, run_name: str, save_dir: str):
    data: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        best = (run.get(split) or {}).get("best", {})
        # also try last epoch from stats if no best.json
        if not best and run.get(split, {}).get("stats"):
            best = run[split]["stats"][-1]
        row = {m: float(best[m]) for m in KEY_METRICS if m in best}
        if row:
            data[split] = row
 
    if not data:
        print("  [WARN] No best-epoch data — skipping bar chart.")
        return
 
    present = [m for m in KEY_METRICS if any(m in v for v in data.values())]
    if not present:
        return
 
    splits  = [s for s in ("train", "val", "test") if s in data]
    x       = np.arange(len(present))
    n_sp    = len(splits)
    w       = 0.22
    offsets = np.linspace(-(n_sp - 1) * w / 2, (n_sp - 1) * w / 2, n_sp)
 
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 2.2), 5),
                            facecolor=THEME["bg"])
    ax.set_facecolor(THEME["panel"])
 
    for sp, off in zip(splits, offsets):
        vals = [data[sp].get(m, 0.0) for m in present]
        bars = ax.bar(
            x + off, vals, w,
            label=sp,
            color=SPLIT_COLOR[sp],
            alpha=0.90,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=6.8, color=THEME["text"],
                    rotation=90,
                )
 
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABEL.get(m, m) for m in present],
        rotation=28, ha="right", fontsize=8.5, color=THEME["text"],
    )
    ax.set_ylabel("Score", fontsize=9, color=THEME["muted"])
    top = min(1.28, (max(
        max(data[sp].get(m, 0) for m in present)
        for sp in splits
    ) + 0.20))
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=7.5, colors=THEME["muted"])
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=THEME["grid"], linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    # Reference lines at 0.5 and 0.75
    for ref in (0.5, 0.75):
        if ref < top:
            ax.axhline(ref, color=THEME["muted"], linewidth=0.5,
                       linestyle=":", alpha=0.6)
 
    # Legend for bar chart: place top-right corner inside (bars don't reach there)
    leg = ax.legend(fontsize=8.5, fancybox=False,
                    edgecolor=THEME["border"], framealpha=0.85,
                    loc="upper right")
    leg.get_frame().set_linewidth(0.5)
 
    fig.suptitle(f"Best-Epoch Metrics  ·  {run_name}",
                 fontsize=11, fontweight="bold",
                 color=THEME["text"], x=0.01, ha="left", y=1.0)
    _save_or_show(fig, save_dir, "02_best_bar.png")


# =============================================================================
# Plot 3 — Timing analysis
# =============================================================================

def plot_timing(run: Dict, run_name: str, save_dir: str):
    if "train" not in run:
        return
    stats = run["train"]["stats"]
    ep,  times = metric_series(stats, "time_epoch")
    _,   etas  = metric_series(stats, "eta_hours")

    if not times:
        print("  [WARN] No timing data — skipping timing plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), facecolor=THEME["bg"])
    col = SPLIT_COLOR["train"]

    # — per-epoch time —
    ax = axes[0]
    ax.set_facecolor(THEME["panel"])
    ax.plot(ep, times, color=col, linewidth=1.2,
            marker="o", markersize=2.2, alpha=0.9)
    # highlight outliers (> mean + 2σ)
    arr = np.array(times)
    thresh = arr.mean() + 2 * arr.std()
    out_x = [e for e, t in zip(ep, times) if t > thresh]
    out_y = [t for t in times if t > thresh]
    if out_x:
        ax.scatter(out_x, out_y, color="#f78166", zorder=5,
                   s=28, label=f"spike (>{thresh:.0f}s)")
        ax.legend(fontsize=7)
    _style_ax(ax, "Train time / epoch  (s)", "seconds")

    # — cumulative time —
    ax = axes[1]
    ax.set_facecolor(THEME["panel"])
    cum = np.cumsum(times) / 60
    ax.fill_between(ep, cum, alpha=0.18, color=col)
    ax.plot(ep, cum, color=col, linewidth=1.4)
    # annotate final value
    ax.annotate(
        f"total: {cum[-1]:.1f} min",
        xy=(ep[-1], cum[-1]),
        xytext=(-55, -18), textcoords="offset points",
        fontsize=8, color=col,
        arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
    )
    _style_ax(ax, "Cumulative time  (min)", "minutes")

    # — ETA curve —
    ax = axes[2]
    ax.set_facecolor(THEME["panel"])
    if etas:
        ep2, etas2 = metric_series(stats, "eta_hours")
        ax.fill_between(ep2, etas2, alpha=0.18, color=SPLIT_COLOR["val"])
        ax.plot(ep2, etas2, color=SPLIT_COLOR["val"], linewidth=1.4)
        _style_ax(ax, "Remaining time  (ETA, hrs)", "hours")
    else:
        ax.text(0.5, 0.5, "no eta data",
                transform=ax.transAxes, ha="center",
                va="center", color=THEME["muted"])
        _style_ax(ax, "ETA", "hours")

    fig.suptitle(f"Training Time  ·  {run_name}",
                 fontsize=11, fontweight="bold",
                 color=THEME["text"], x=0.01, ha="left", y=1.01)
    _save_or_show(fig, save_dir, "03_timing.png")


# =============================================================================
# Plot 4 — Per-repeat overlay (shows variance across seeds)
# =============================================================================

def plot_repeat_variance(run_dir: str, metric: str, split: str,
                         run_name: str, save_dir: str):
    """
    Overlay all repeat curves for one metric on one split.
    Shows variance across random seeds — useful for small datasets.
    """
    repeats = find_repeat_dirs(run_dir)
    if len(repeats) <= 1:
        return

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["panel"])

    palette = plt.cm.cool(np.linspace(0.15, 0.85, len(repeats)))
    all_vals: List[np.ndarray] = []
    all_ep:   List[int]        = []

    for i, rep_dir in enumerate(repeats):
        split_dir = osp.join(rep_dir, split)
        stats     = _load_stats(osp.join(split_dir, "stats.json"))
        ep, vals  = metric_series(stats, metric)
        if not vals:
            continue
        ax.plot(ep, vals, color=palette[i], linewidth=1.0,
                alpha=0.55, label=f"repeat {osp.basename(rep_dir)}")
        all_vals.append(np.array(vals))
        all_ep = ep

    # Mean ± std band
    if len(all_vals) > 1:
        min_len = min(len(v) for v in all_vals)
        mat     = np.stack([v[:min_len] for v in all_vals])
        mean    = mat.mean(0)
        std     = mat.std(0)
        ep_trim = all_ep[:min_len]
        ax.plot(ep_trim, mean, color="white", linewidth=2.0,
                zorder=5, label="mean")
        ax.fill_between(ep_trim, mean - std, mean + std,
                        color="white", alpha=0.12, zorder=4, label="±1σ")

    _style_ax(ax,
              f"Repeat Variance  ·  {METRIC_LABEL.get(metric, metric)}  [{split}]",
              METRIC_LABEL.get(metric, metric))
    fig.suptitle(f"Seed Variance  ·  {run_name}",
                 fontsize=11, fontweight="bold",
                 color=THEME["text"], x=0.01, ha="left", y=1.01)
    fname = f"04_variance_{split}_{metric.replace('-','_')}.png"
    _save_or_show(fig, save_dir, fname)


# =============================================================================
# Plot 5 — Multi-run comparison
# =============================================================================

def plot_comparison(runs: Dict[str, Dict], metrics: List[str],
                    split: str, save_dir: str):
    present = [
        m for m in metrics
        if any(metric_series(rd.get(split, {}).get("stats", []), m)[1]
               for rd in runs.values())
    ]
    if not present:
        print(f"  [WARN] No data for comparison on '{split}' split.")
        return

    n    = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5),
                              facecolor=THEME["bg"])
    if n == 1:
        axes = [axes]

    palette = [
        "#58a6ff", "#f0883e", "#3fb950", "#d2a8ff",
        "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
    ]

    for ax, metric in zip(axes, present):
        ax.set_facecolor(THEME["panel"])
        for i, (name, rd) in enumerate(runs.items()):
            ep, vals = metric_series(
                rd.get(split, {}).get("stats", []), metric
            )
            if not vals:
                continue
            c = palette[i % len(palette)]
            ax.plot(ep, vals, color=c, linewidth=1.5,
                    label=name, alpha=0.9)

        _style_ax(ax,
                  f"{METRIC_LABEL.get(metric, metric)}  [{split}]",
                  METRIC_LABEL.get(metric, metric))

    fig.suptitle(f"Run Comparison  ·  {split} split",
                 fontsize=11, fontweight="bold",
                 color=THEME["text"], x=0.01, ha="left", y=1.01)
    _save_or_show(fig, save_dir, f"05_compare_{split}.png")


# =============================================================================
# Console summary
# =============================================================================

def print_best(run: Dict, run_name: str):
    print(f"\n{SEP2}")
    print(f"  {run_name}")
    print(SEP2)
    row_keys = ["loss"] + KEY_METRICS + ["lr"]

    for split in ("train", "val", "test"):
        sd   = run.get(split, {})
        best = sd.get("best", {})
        # fallback: last row in stats
        if not best and sd.get("stats"):
            best = sd["stats"][-1]
        if not best:
            continue

        ep = best.get("epoch", "?")
        print(f"\n  [{split.upper()}]  best epoch = {ep}")
        print(f"  {SEP[:60]}")
        for k in row_keys:
            if k not in best:
                continue
            label = METRIC_LABEL.get(k, k)
            val   = best[k]
            bar   = ""
            if isinstance(val, float) and 0.0 <= val <= 1.0 and k != "lr":
                filled = int(val * 30)
                bar    = "  " + "█" * filled + "░" * (30 - filled)
            print(f"  {label:<26}  {val:.6f}{bar}")


def print_comparison_table(runs: Dict[str, Dict]):
    print(f"\n{SEP2}")
    print("  COMPARISON TABLE  (best-epoch values)")
    print(SEP2)

    col_w = 14
    for split in ("val", "test"):
        print(f"\n  {split.upper()} SPLIT")
        header = f"  {'run':<32}" + "".join(
            f"{METRIC_LABEL.get(m, m)[:col_w-1]:<{col_w}}"
            for m in KEY_METRICS
        )
        print(header)
        print(f"  {SEP}")
        for name, rd in runs.items():
            best = (rd.get(split) or {}).get("best", {})
            if not best and rd.get(split, {}).get("stats"):
                best = rd[split]["stats"][-1]
            row = f"  {name:<32}"
            for m in KEY_METRICS:
                v = best.get(m)
                cell = f"{v:.4f}" if v is not None else "—"
                row += f"{cell:<{col_w}}"
            print(row)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Visualise GraphGPS / JSLibs training results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",   default="results/jslibs-v3",
                   help="Run directory (contains 0/ repeat dirs + agg/)")
    p.add_argument("--compare",   nargs="+", default=[], metavar="NAME",
                   help="Sub-dir names inside --run_dir to compare")
    p.add_argument("--split",     default="val",
                   choices=["train", "val", "test"],
                   help="Split used for comparison plot")
    p.add_argument("--metrics",   nargs="+", default=[], metavar="METRIC",
                   help="Metrics for comparison (default: accuracy accuracy-JS f1 auc)")
    p.add_argument("--save_dir",  default="",
                   help="Directory for PNG output (empty = show interactively)")
    p.add_argument("--variance",  action="store_true",
                   help="Plot per-repeat variance overlay")
    p.add_argument("--no_timing", action="store_true",
                   help="Skip timing plot")
    args = p.parse_args()

    if args.save_dir:
        matplotlib.use("Agg")

    _apply_theme()

    # ── comparison mode ───────────────────────────────────────────────────────
    if args.compare:
        runs: Dict[str, Dict] = {}
        for name in args.compare:
            rdir = osp.join(args.run_dir, name)
            if not osp.isdir(rdir):
                print(f"  [WARN] Not found: {rdir}")
                continue
            rd = load_run(rdir)
            if rd:
                runs[name] = rd
            else:
                print(f"  [WARN] No data in {rdir}")

        if not runs:
            sys.exit("[ERROR] No valid runs found for comparison.")

        print_comparison_table(runs)

        cmp_m = args.metrics or [
            m for m in ("accuracy", "accuracy-JS", "f1", "auc")
            if any(
                metric_series(rd.get(args.split, {}).get("stats", []), m)[1]
                for rd in runs.values()
            )
        ] or ["accuracy", "loss"]

        for sp in ("val", "test"):
            plot_comparison(runs, cmp_m, sp, args.save_dir)
        print(f"\n{SEP2}\n  Done.\n{SEP2}\n")
        return

    # ── single-run mode ───────────────────────────────────────────────────────
    if not osp.isdir(args.run_dir):
        sys.exit(f"[ERROR] Not found: {args.run_dir}")

    run      = load_run(args.run_dir)
    run_name = osp.basename(args.run_dir.rstrip("/"))

    if not run:
        sys.exit(f"[ERROR] No stats data found in {args.run_dir}")

    print_best(run, run_name)

    plot_curves(run, run_name, args.save_dir)
    plot_best_bar(run, run_name, args.save_dir)

    if not args.no_timing:
        plot_timing(run, run_name, args.save_dir)

    if args.variance:
        for m in ("accuracy", "accuracy-JS", "loss"):
            if any(metric_series(run.get("val", {}).get("stats", []), m)[1]):
                plot_repeat_variance(
                    args.run_dir, m, "val", run_name, args.save_dir
                )

    print(f"\n{SEP2}")
    print("  Done.")
    if args.save_dir:
        print(f"  Figures → {osp.abspath(args.save_dir)}")
    print(f"{SEP2}\n")


if __name__ == "__main__":
    main()