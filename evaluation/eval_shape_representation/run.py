"""
Shape Representation Analysis — HDR vs Point-Fraction

Three representative pairs for chr1:
  - UNTR 12h vs UNTR 18h  : stability (same condition, adjacent timepoints)
  - VACV 12h vs VACV 18h  : temporal change (infected condition)
  - UNTR vs VACV at 18h   : condition difference (same timepoint)

Settings: PCA+ICP alignment, YZ plane, 60% HDR/PF level.
Compares IoU and meanNN between HDR and point-fraction representations.

Usage:
    python run.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import mpase

DATA_ROOT   = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

CHROM    = "chr1"
PLANE    = "YZ"
LEVEL    = 60
XYZ_COLS = ("middle_x", "middle_y", "middle_z")

CFG_COMMON = mpase.CfgCommon(icp_iters=30)
CFG_HDR    = mpase.CfgHDR(n_boot=128, mass_levels=(0.60, 0.80, 0.95, 1.00))
CFG_PF     = mpase.CfgPF(frac_levels=(0.60, 0.80, 0.95, 1.00))

def _gene_info(chrom, hrs, cond):
    return os.path.join(DATA_ROOT, chrom, hrs, cond,
                        f"structure_{hrs}_{cond}_gene_info.csv")

PAIRS = [
    {
        "name":    "UNTR 12h vs 18h",
        "label":   "Stability",
        "csv_a":   _gene_info(CHROM, "12hrs", "untr"),
        "csv_b":   _gene_info(CHROM, "18hrs", "untr"),
        "label_a": "untr_12h",
        "label_b": "untr_18h",
    },
    {
        "name":    "VACV 12h vs 18h",
        "label":   "Temporal change",
        "csv_a":   _gene_info(CHROM, "12hrs", "vacv"),
        "csv_b":   _gene_info(CHROM, "18hrs", "vacv"),
        "label_a": "vacv_12h",
        "label_b": "vacv_18h",
    },
    {
        "name":    "UNTR vs VACV at 18h",
        "label":   "Condition difference",
        "csv_a":   _gene_info(CHROM, "18hrs", "untr"),
        "csv_b":   _gene_info(CHROM, "18hrs", "vacv"),
        "label_a": "untr_18h",
        "label_b": "vacv_18h",
    },
]


def extract_metric(result: dict, plane: str, level: int, variant: str) -> tuple:
    df = result["metrics"]
    row = df[(df["plane"] == plane) & (df["level"] == level) & (df["variant"] == variant)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["IoU"]), float(row.iloc[0]["meanNN"])


def run_all() -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        print(f"\n  {pair['name']} ({pair['label']})")
        result = mpase.run(
            csv_list=[pair["csv_a"], pair["csv_b"]],
            labels=[pair["label_a"], pair["label_b"]],
            xyz_cols=XYZ_COLS,
            cfg_common=CFG_COMMON,
            cfg_hdr=CFG_HDR,
            cfg_pf=CFG_PF,
        )
        for variant, label in [("hdr", "HDR"), ("point_fraction", "PF")]:
            iou, meannn = extract_metric(result, PLANE, LEVEL, variant)
            rows.append({
                "pair":            pair["name"],
                "category":        pair["label"],
                "representation":  label,
                "plane":           PLANE,
                "level":           LEVEL,
                "IoU":             round(iou, 4),
                "meanNN":          round(meannn, 3),
            })
            print(f"    {label:4}  IoU={iou:.3f}  meanNN={meannn:.2f}")
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def plot_grouped_bars(df: pd.DataFrame):
    """Side-by-side grouped bar: IoU and meanNN for HDR vs PF across pairs."""
    pair_order = [p["name"] for p in PAIRS]
    palette    = {"HDR": "#4c72b0", "PF": "#dd8452"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        x     = np.arange(len(pair_order))
        width = 0.35
        for i, rep in enumerate(["HDR", "PF"]):
            vals = [df[(df["pair"] == p) & (df["representation"] == rep)][metric].values
                    for p in pair_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=rep, color=palette[rep], alpha=0.88)
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi + (i - 0.5) * width, v + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_order, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — HDR vs PF (YZ, 60%)")
        ax.legend()
        if metric == "IoU":
            ax.set_ylim(0, 0.7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Shape representation comparison — Chr1", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "grouped_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_heatmaps(df: pd.DataFrame):
    """Two heatmaps: pairs × representations for IoU and meanNN."""
    pair_order = [p["name"] for p in PAIRS]

    for metric, cmap, vmin, vmax in [
        ("IoU",    "RdYlGn",   0, 1),
        ("meanNN", "RdYlGn_r", 0, 20),
    ]:
        pivot = df.pivot_table(index="pair", columns="representation",
                               values=metric, aggfunc="first")
        pivot = pivot.reindex(pair_order)[["HDR", "PF"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=ax,
                    linewidths=0.8, cbar_kws={"label": metric})
        title_note = "lower = better" if metric == "meanNN" else "higher = better"
        ax.set_title(f"{metric} — HDR vs PF\n(YZ, 60%, Chr1)  [{title_note}]")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"heatmap_{metric.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_delta(df: pd.DataFrame):
    """Bar chart showing IoU difference (PF - HDR) per pair."""
    pair_order = [p["name"] for p in PAIRS]
    categories = [p["label"] for p in PAIRS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        deltas = []
        for p in pair_order:
            hdr = df[(df["pair"] == p) & (df["representation"] == "HDR")][metric].values
            pf  = df[(df["pair"] == p) & (df["representation"] == "PF")][metric].values
            deltas.append(float(pf[0] - hdr[0]) if len(hdr) and len(pf) else np.nan)

        bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        bars = ax.bar(range(len(pair_order)), deltas, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(pair_order, categories)],
                           rotation=10, ha="right", fontsize=8)
        ax.set_ylabel(f"Δ {metric}  (PF − HDR)")
        direction = "↑ PF better" if metric == "IoU" else "↓ PF better"
        ax.set_title(f"Δ {metric} (PF − HDR)\n{direction}")
        for xi, v in enumerate(deltas):
            if not np.isnan(v):
                ax.text(xi, v + (0.001 if v >= 0 else -0.003),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("PF vs HDR difference — Chr1 (YZ, 60%)", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "delta_pf_vs_hdr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Running shape representation analysis...")
    df = run_all()

    out_csv = os.path.join(RESULTS_DIR, "shape_representation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    print("\nGenerating figures...")
    plot_grouped_bars(df)
    plot_heatmaps(df)
    plot_delta(df)

    print("\n── Results ───────────────────────────────────────")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
