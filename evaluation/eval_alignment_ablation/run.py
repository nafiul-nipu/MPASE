"""
Alignment Ablation — Real Data (UNTR 12h vs UNTR 18h, all chromosomes)

Compares three alignment modes:
  none    : align_mode="skip"
  pca     : align_mode="auto", icp_iters=0  (PCA pre-align, no ICP)
  pca_icp : align_mode="auto", icp_iters=30 (full pipeline)

We use only UNTR 12h vs UNTR 18h (same condition, adjacent timepoints) across
all chromosomes. Keeping condition fixed isolates the alignment contribution
from biological variation.

Figures are saved under figures/{hdr|pf}/level_{60|80|95|100}/.

Usage:
    python run.py
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import mpase

# ── paths ──────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

CHROMS = [
    "chr1",
    "chr2","chr3","chr4","chr5","chr6","chr7",
    "chr10","chr12","chr13","chr14","chr15","chr16",
    "chr18","chr19","chr20","chr21","chr22","chr23",
    "chr25","chr26","chr27","chr28","chr29",
]

ALIGN_MODES = {
    "none":    dict(align_mode="skip", cfg_common=mpase.CfgCommon()),
    "pca":     dict(align_mode="auto", cfg_common=mpase.CfgCommon(icp_iters=0)),
    "pca_icp": dict(align_mode="auto", cfg_common=mpase.CfgCommon(icp_iters=30)),
}

PLANE    = "YZ"
LEVELS   = [60, 80, 95, 100]
VARIANTS = [
    ("hdr",            "hdr"),
    ("point_fraction", "pf"),
]
TIMES = ["12hrs", "18hrs", "24hrs"]
CONDS = ["untr", "vacv"]

XYZ_COLS = ("middle_x", "middle_y", "middle_z")

CFG_HDR = mpase.CfgHDR(n_boot=256, mass_levels=(0.60, 0.80, 0.95, 1.00))
CFG_PF  = mpase.CfgPF(frac_levels=(0.60, 0.80, 0.95, 1.00))


def collect_csvs_labels(chrom: str):
    csvs, labels = [], []
    for hrs in TIMES:
        for cond in CONDS:
            p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                             f"structure_{hrs}_{cond}_gene_info.csv")
            if os.path.exists(p):
                csvs.append(p)
                labels.append(f"{chrom}_{hrs}_{cond}")
    return csvs, labels


def extract_metric(result: dict, plane: str, level: int, variant: str,
                   A_label: str, B_label: str) -> tuple:
    df = result["metrics"]
    row = df[(df["plane"] == plane) & (df["level"] == level) &
             (df["variant"] == variant) &
             (df["A"] == A_label) & (df["B"] == B_label)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["IoU"]), float(row.iloc[0]["meanNN"])


def run_all() -> pd.DataFrame:
    rows = []
    for chrom in CHROMS:
        csvs, labels = collect_csvs_labels(chrom)
        if len(csvs) < 2:
            print(f"  {chrom}: missing files, skipping")
            continue

        A_label = f"{chrom}_12hrs_untr"
        B_label = f"{chrom}_18hrs_untr"

        for mode_name, mode_kwargs in ALIGN_MODES.items():
            try:
                result = mpase.run(
                    csv_list=csvs,
                    labels=labels,
                    xyz_cols=XYZ_COLS,
                    cfg_hdr=CFG_HDR,
                    cfg_pf=CFG_PF,
                    **mode_kwargs,
                )
                for variant_key, variant_folder in VARIANTS:
                    for level in LEVELS:
                        iou, meannn = extract_metric(result, PLANE, level, variant_key,
                                                     A_label, B_label)
                        rows.append({
                            "chrom":          chrom,
                            "align":          mode_name,
                            "variant":        variant_key,
                            "variant_folder": variant_folder,
                            "plane":          PLANE,
                            "level":          level,
                            "IoU":            round(iou, 4),
                            "meanNN":         round(meannn, 3),
                        })
                print(f"  {chrom:6}  {mode_name:8}  done")
            except Exception as e:
                print(f"  {chrom:6}  {mode_name:8}  ERROR: {e}")

    return pd.DataFrame(rows)


def find_best_cases(df: pd.DataFrame, top_n: int = 6) -> list:
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")
    if not {"none", "pca", "pca_icp"}.issubset(pivot.columns):
        return df["chrom"].unique().tolist()
    pivot["gain"]     = pivot["pca_icp"] - pivot["none"]
    pivot["monotone"] = (pivot["pca"] >= pivot["none"]) & (pivot["pca_icp"] >= pivot["pca"])
    best = pivot[pivot["monotone"]].sort_values("gain", ascending=False)
    return best.head(top_n).index.tolist()


# ── figures ─────────────────────────────────────────────────────────────────

def fig_dir(variant_folder: str, level: int) -> str:
    d = os.path.join(FIGURES_DIR, variant_folder, f"level_{level}")
    os.makedirs(d, exist_ok=True)
    return d


def plot_all_chromosomes(df: pd.DataFrame, best_chroms: list, variant_folder: str, level: int):
    order   = ["none", "pca", "pca_icp"]
    palette = {"none": "#d9534f", "pca": "#f0ad4e", "pca_icp": "#5cb85c"}
    labels  = {"none": "No alignment", "pca": "PCA only", "pca_icp": "PCA + ICP"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        chrom_order = sorted(df["chrom"].unique(), key=lambda c: int(c.replace("chr", "")))
        x     = np.arange(len(chrom_order))
        width = 0.26
        for i, mode in enumerate(order):
            vals = [df[(df["chrom"] == c) & (df["align"] == mode)][metric].values for c in chrom_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 1) * width, vals, width, label=labels[mode], color=palette[mode], alpha=0.85)
            for j, (c, v) in enumerate(zip(chrom_order, vals)):
                if c in best_chroms and not np.isnan(v):
                    ax.bar(x[j] + (i - 1) * width, v, width,
                           color=palette[mode], alpha=1.0, edgecolor="black", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by alignment method — all chromosomes\n(bold = best-case)")
        ax.legend()
        if metric == "IoU":
            ax.set_ylim(0, 1)
        ax.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "all_chroms_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_best_cases(df: pd.DataFrame, best_chroms: list, variant_folder: str, level: int):
    order   = ["none", "pca", "pca_icp"]
    xlabels = ["No alignment", "PCA only", "PCA + ICP"]
    sub     = df[df["chrom"].isin(best_chroms)]
    palette = sns.color_palette("tab10", len(best_chroms))

    for metric, fname in [("IoU", "trend_iou.png"), ("meanNN", "trend_meannn.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for color, chrom in zip(palette, best_chroms):
            vals = []
            for mode in order:
                row = sub[(sub["chrom"] == chrom) & (sub["align"] == mode)][metric]
                vals.append(row.values[0] if len(row) else np.nan)
            ax.plot(range(3), vals, marker="o", label=chrom, color=color, linewidth=2)
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (xi, v),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7, color=color)
        ax.set_xticks(range(3))
        ax.set_xticklabels(xlabels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — UNTR 12h vs 18h (YZ, {level}%)")
        ax.legend(fontsize=8)
        if metric == "IoU":
            ax.set_ylim(0, 0.6)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(fig_dir(variant_folder, level), fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_gain_heatmap(df: pd.DataFrame, variant_folder: str, level: int):
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")
    pivot = pivot[["none", "pca", "pca_icp"]].copy()
    pivot.columns = ["No alignment", "PCA only", "PCA + ICP"]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))
    fig, ax = plt.subplots(figsize=(7, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, cbar_kws={"label": "IoU"})
    ax.set_title(f"IoU by alignment method\n(UNTR 12h vs 18h, YZ {level}%)")
    ax.set_xlabel("")
    ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "gain_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_meannn_heatmap(df: pd.DataFrame, variant_folder: str, level: int):
    pivot = df.pivot_table(index="chrom", columns="align", values="meanNN")
    pivot = pivot[["none", "pca", "pca_icp"]].copy()
    pivot.columns = ["No alignment", "PCA only", "PCA + ICP"]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))
    vmax  = 20
    annot = pivot.round(2).astype(str)
    fig, ax = plt.subplots(figsize=(7, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn_r",
                vmin=0, vmax=vmax, ax=ax,
                linewidths=0.5, cbar_kws={"label": "meanNN (pixels, capped at 20)"})
    ax.set_title(f"meanNN by alignment method\n(UNTR 12h vs 18h, YZ {level}%)\nlower = better")
    ax.set_xlabel("")
    ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "meannn_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_all_chroms_trend(df: pd.DataFrame, variant_folder: str, level: int):
    order   = ["none", "pca", "pca_icp"]
    xlabels = ["No alignment", "PCA only", "PCA + ICP"]
    chroms  = sorted(df["chrom"].unique(), key=lambda c: int(c.replace("chr", "")))
    palette = sns.color_palette("tab20", len(chroms))

    for metric, fname in [("IoU", "all_trend_iou.png"), ("meanNN", "all_trend_meannn.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for color, chrom in zip(palette, chroms):
            sub  = df[df["chrom"] == chrom]
            vals = [sub[sub["align"] == mode][metric].values for mode in order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.plot(range(3), vals, marker="o", label=chrom, color=color, linewidth=1.5, alpha=0.85)
        ax.set_xticks(range(3))
        ax.set_xticklabels(xlabels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — all chromosomes (UNTR 12h vs 18h, YZ {level}%)")
        ax.legend(fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(1, 1))
        if metric == "IoU":
            ax.set_ylim(0, 0.6)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(fig_dir(variant_folder, level), fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_iou_gain_heatmap(df: pd.DataFrame, variant_folder: str, level: int, clip: float = 0.15):
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")
    if not {"none", "pca", "pca_icp"}.issubset(pivot.columns):
        return
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))

    gains = pd.DataFrame(index=pivot.index)
    gains["PCA − None"]     = pivot["pca"]     - pivot["none"]
    gains["PCA+ICP − None"] = pivot["pca_icp"] - pivot["none"]

    clipped = gains.clip(-clip, clip)
    vmax    = clip
    annot   = gains.round(3).astype(str)

    variant_label = "HDR" if variant_folder == "hdr" else "PF"
    fig, ax = plt.subplots(figsize=(5, max(5, len(gains) * 0.35)))
    sns.heatmap(clipped, annot=annot, fmt="", cmap="RdBu",
                vmin=-vmax, vmax=vmax, center=0,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": f"IoU gain (clipped ±{clip})"})
    ax.set_title(f"IoU Gain from Alignment — {variant_label}\n"
                 f"(UNTR 12h vs 18h, YZ {level}%)\n"
                 f"positive = improvement, negative = degradation")
    ax.set_xlabel("")
    ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "iou_gain_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def generate_figures_for(df: pd.DataFrame, variant_key: str, variant_folder: str, level: int):
    sub = df[(df["variant"] == variant_key) & (df["level"] == level)].copy()
    if sub.empty:
        return
    best_chroms = find_best_cases(sub)
    plot_all_chromosomes(sub, best_chroms, variant_folder, level)
    plot_best_cases(sub, best_chroms, variant_folder, level)
    plot_gain_heatmap(sub, variant_folder, level)
    plot_meannn_heatmap(sub, variant_folder, level)
    plot_all_chroms_trend(sub, variant_folder, level)
    plot_iou_gain_heatmap(sub, variant_folder, level)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running alignment ablation (UNTR 12h vs 18h, all chromosomes)...")
    df = run_all()

    if df.empty:
        print("No results — check data paths.")
        return

    full_csv = os.path.join(RESULTS_DIR, "ablation_all.csv")
    df.to_csv(full_csv, index=False)
    print(f"\nSaved {full_csv}")

    print("\nGenerating figures...")
    for variant_key, variant_folder in VARIANTS:
        for level in LEVELS:
            print(f"\n  [{variant_folder.upper()}] level {level}%")
            generate_figures_for(df, variant_key, variant_folder, level)

    print("\nDone.")


if __name__ == "__main__":
    main()
