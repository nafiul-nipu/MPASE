"""
Alignment Ablation — Real Data (UNTR 12h vs 18h, all chromosomes)

Compares four alignment modes:
  none        : align_mode="skip"
  naive_pca   : PCA axes aligned directly — no axis permutation or sign
                disambiguation (can produce mirror images / swapped axes)
  pca         : align_mode="auto", icp_iters=0  (PCA with full disambiguation, no ICP)
  pca_icp     : align_mode="auto", icp_iters=30 (full MPASE pipeline)

We use only UNTR 12h vs 18h (same condition, adjacent timepoints) across
all chromosomes. Keeping condition fixed isolates the alignment contribution
from biological variation.

Figures are saved under figures/{hdr|pf}/level_{60|80|95|100}/.

Usage:
    source venv/bin/activate
    python -u evaluation/eval_alignment_ablation/run.py 2>&1 | tee /tmp/align_ablation.log
"""

import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import mpase
from mpase.point_alignment import pca_axes

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


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── naive PCA alignment ───────────────────────────────────────────────────────

def _naive_pca_align(raw_arrays):
    """
    Align each set to set[0] using raw PCA axes with no disambiguation.
    PCA axes have sign and permutation ambiguity; without resolving these
    the result can be mirrored or axis-swapped.
    """
    ref = raw_arrays[0] - raw_arrays[0].mean(0)
    Va  = pca_axes(ref)
    aligned = [ref]
    for pts in raw_arrays[1:]:
        centered = pts - pts.mean(0)
        Vb = pca_axes(centered)
        R  = Va @ Vb.T        # direct axis match — no disambiguation
        aligned.append(centered @ R.T)
    return aligned


# ── data loading ──────────────────────────────────────────────────────────────

def collect_csvs_labels(chrom: str):
    csvs, labels, arrays = [], [], []
    for hrs in TIMES:
        for cond in CONDS:
            p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                             f"structure_{hrs}_{cond}_gene_info.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)[list(XYZ_COLS)].dropna()
                csvs.append(p)
                labels.append(f"{chrom}_{hrs}_{cond}")
                arrays.append(df.values.astype(np.float32))
    return csvs, labels, arrays


# ── run one mode ──────────────────────────────────────────────────────────────

def run_mode(mode, csvs, labels, arrays):
    if mode == "none":
        return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                         cfg_hdr=CFG_HDR, cfg_pf=CFG_PF,
                         align_mode="skip")

    if mode == "naive_pca":
        aligned = _naive_pca_align(arrays)
        return mpase.run(points_list=aligned, labels=labels,
                         cfg_hdr=CFG_HDR, cfg_pf=CFG_PF,
                         align_mode="skip")

    if mode == "pca":
        return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                         cfg_hdr=CFG_HDR, cfg_pf=CFG_PF,
                         align_mode="auto",
                         cfg_common=mpase.CfgCommon(icp_iters=0))

    if mode == "pca_icp":
        return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                         cfg_hdr=CFG_HDR, cfg_pf=CFG_PF,
                         align_mode="auto",
                         cfg_common=mpase.CfgCommon(icp_iters=30))

    raise ValueError(f"Unknown mode: {mode}")


ALIGN_MODES = {
    "none":      dict(),
    "naive_pca": dict(),
    "pca":       dict(),
    "pca_icp":   dict(),
}


# ── metrics extraction ────────────────────────────────────────────────────────

def extract_metric(result: dict, plane: str, level: int, variant: str,
                   A_label: str, B_label: str) -> tuple:
    df = result["metrics"]
    row = df[(df["plane"] == plane) & (df["level"] == level) &
             (df["variant"] == variant) &
             (df["A"] == A_label) & (df["B"] == B_label)]
    if row.empty:
        row = df[(df["plane"] == plane) & (df["level"] == level) &
                 (df["variant"] == variant) &
                 (df["A"] == B_label) & (df["B"] == A_label)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["IoU"]), float(row.iloc[0]["meanNN"])


# ── main run ──────────────────────────────────────────────────────────────────

def run_all() -> pd.DataFrame:
    rows = []
    for chrom in CHROMS:
        csvs, labels, arrays = collect_csvs_labels(chrom)
        if len(csvs) < 2:
            _log(f"  {chrom}: missing files, skipping")
            continue

        A_label = f"{chrom}_12hrs_untr"
        B_label = f"{chrom}_18hrs_untr"

        _log(f"  {chrom}")
        for mode_name in ALIGN_MODES:
            t0 = time.time()
            try:
                result = run_mode(mode_name, csvs, labels, arrays)
                for variant_key, variant_folder in VARIANTS:
                    for level in LEVELS:
                        iou, meannn = extract_metric(result, PLANE, level,
                                                     variant_key, A_label, B_label)
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
                _log(f"    {mode_name:12}  {time.time()-t0:.1f}s  done")
            except Exception as e:
                _log(f"    {mode_name:12}  ERROR: {e}")

    return pd.DataFrame(rows)


# ── figures (same as original + naive_pca) ───────────────────────────────────

def find_best_cases(df: pd.DataFrame, top_n: int = 6) -> list:
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")
    needed = {"none", "pca", "pca_icp"}
    if not needed.issubset(pivot.columns):
        return df["chrom"].unique().tolist()
    pivot["gain"]     = pivot["pca_icp"] - pivot["none"]
    pivot["monotone"] = (pivot.get("naive_pca", pivot["none"]) >= pivot["none"]) & \
                        (pivot["pca"] >= pivot.get("naive_pca", pivot["none"])) & \
                        (pivot["pca_icp"] >= pivot["pca"])
    best = pivot[pivot["monotone"]].sort_values("gain", ascending=False)
    return best.head(top_n).index.tolist()


def fig_dir(variant_folder: str, level: int) -> str:
    d = os.path.join(FIGURES_DIR, variant_folder, f"level_{level}")
    os.makedirs(d, exist_ok=True)
    return d


ORDER   = ["none", "naive_pca", "pca", "pca_icp"]
PALETTE = {
    "none":      "#d9534f",
    "naive_pca": "#f0ad4e",
    "pca":       "#5bc0de",
    "pca_icp":   "#5cb85c",
}
XLABELS = {
    "none":      "No alignment",
    "naive_pca": "Naive PCA",
    "pca":       "PCA + Disambig",
    "pca_icp":   "PCA + ICP",
}


def plot_all_chromosomes(df: pd.DataFrame, best_chroms: list,
                         variant_folder: str, level: int):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        chrom_order = sorted(df["chrom"].unique(),
                             key=lambda c: int(c.replace("chr", "")))
        x     = np.arange(len(chrom_order))
        n     = len(ORDER)
        width = 0.8 / n
        for i, mode in enumerate(ORDER):
            sub  = df[df["align"] == mode]
            vals = [sub[sub["chrom"] == c][metric].values for c in chrom_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - (n - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=XLABELS[mode], color=PALETTE[mode], alpha=0.85)
            for j, (c, v) in enumerate(zip(chrom_order, vals)):
                if c in best_chroms and not np.isnan(v):
                    ax.bar(x[j] + offset, v, width,
                           color=PALETTE[mode], alpha=1.0,
                           edgecolor="black", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by alignment method — all chromosomes\n"
                     f"(bold = monotone improvement cases)")
        ax.legend()
        if metric == "IoU":
            ax.set_ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "all_chroms_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def plot_best_cases(df: pd.DataFrame, best_chroms: list,
                    variant_folder: str, level: int):
    sub     = df[df["chrom"].isin(best_chroms)]
    palette = sns.color_palette("tab10", len(best_chroms))
    x       = np.arange(len(ORDER))

    for metric, fname in [("IoU", "trend_iou.png"), ("meanNN", "trend_meannn.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for color, chrom in zip(palette, best_chroms):
            vals = []
            for mode in ORDER:
                row = sub[(sub["chrom"] == chrom) & (sub["align"] == mode)][metric]
                vals.append(row.values[0] if len(row) else np.nan)
            ax.plot(x, vals, marker="o", label=chrom, color=color, linewidth=2)
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (xi, v),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([XLABELS[k] for k in ORDER])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — UNTR 12h vs 18h (YZ, {level}%)\nbest-case chromosomes")
        ax.legend(fontsize=8)
        if metric == "IoU":
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(fig_dir(variant_folder, level), fname)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        _log(f"  Saved {out}")


def plot_gain_heatmap(df: pd.DataFrame, variant_folder: str, level: int):
    modes_present = [m for m in ORDER if m in df["align"].unique()]
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")[modes_present]
    pivot.columns = [XLABELS[k] for k in modes_present]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, cbar_kws={"label": "IoU"})
    ax.set_title(f"IoU by alignment method\n(UNTR 12h vs 18h, YZ {level}%)")
    ax.set_xlabel(""); ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "gain_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def plot_meannn_heatmap(df: pd.DataFrame, variant_folder: str, level: int):
    modes_present = [m for m in ORDER if m in df["align"].unique()]
    pivot = df.pivot_table(index="chrom", columns="align", values="meanNN")[modes_present]
    pivot.columns = [XLABELS[k] for k in modes_present]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))
    vmax  = 20
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=pivot.round(2).astype(str), fmt="",
                cmap="RdYlGn_r", vmin=0, vmax=vmax, ax=ax,
                linewidths=0.5, cbar_kws={"label": "meanNN (pixels, capped 20)"})
    ax.set_title(f"meanNN by alignment method\n(UNTR 12h vs 18h, YZ {level}%) — lower = better")
    ax.set_xlabel(""); ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "meannn_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def plot_all_chroms_trend(df: pd.DataFrame, variant_folder: str, level: int):
    chroms  = sorted(df["chrom"].unique(), key=lambda c: int(c.replace("chr", "")))
    palette = sns.color_palette("tab20", len(chroms))
    x       = np.arange(len(ORDER))

    for metric, fname in [("IoU", "all_trend_iou.png"), ("meanNN", "all_trend_meannn.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for color, chrom in zip(palette, chroms):
            sub  = df[df["chrom"] == chrom]
            vals = [sub[sub["align"] == mode][metric].values for mode in ORDER]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.plot(x, vals, marker="o", label=chrom, color=color,
                    linewidth=1.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([XLABELS[k] for k in ORDER])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — all chromosomes (UNTR 12h vs 18h, YZ {level}%)")
        ax.legend(fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(1, 1))
        if metric == "IoU":
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(fig_dir(variant_folder, level), fname)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        _log(f"  Saved {out}")


def plot_iou_gain_heatmap(df: pd.DataFrame, variant_folder: str, level: int,
                          clip: float = 0.15):
    pivot = df.pivot_table(index="chrom", columns="align", values="IoU")
    if not {"none", "pca_icp"}.issubset(pivot.columns):
        return
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))

    gains = pd.DataFrame(index=pivot.index)
    if "naive_pca" in pivot.columns:
        gains["Naive PCA − None"]    = pivot["naive_pca"] - pivot["none"]
    if "pca" in pivot.columns:
        gains["PCA+Disambig − None"] = pivot["pca"]       - pivot["none"]
    gains["PCA+ICP − None"]          = pivot["pca_icp"]   - pivot["none"]

    clipped = gains.clip(-clip, clip)
    variant_label = "HDR" if variant_folder == "hdr" else "PF"
    fig, ax = plt.subplots(figsize=(6, max(5, len(gains) * 0.35)))
    sns.heatmap(clipped, annot=gains.round(3).astype(str), fmt="",
                cmap="RdBu", vmin=-clip, vmax=clip, center=0,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": f"IoU gain vs No Alignment (clipped ±{clip})"})
    ax.set_title(f"IoU Gain from Alignment — {variant_label}\n"
                 f"(UNTR 12h vs 18h, YZ {level}%)")
    ax.set_xlabel(""); ax.set_ylabel("Chromosome")
    plt.tight_layout()
    out = os.path.join(fig_dir(variant_folder, level), "iou_gain_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def generate_figures_for(df: pd.DataFrame, variant_key: str,
                         variant_folder: str, level: int):
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
    os.makedirs(FIGURES_DIR, exist_ok=True)

    _log("Alignment ablation — 4 modes × all chromosomes × 4 levels (UNTR 12h vs 18h)")
    t0 = time.time()

    df = run_all()

    if df.empty:
        _log("No results — check data paths.")
        return

    out_csv = os.path.join(RESULTS_DIR, "ablation_all.csv")
    df.to_csv(out_csv, index=False)
    _log(f"\nSaved {out_csv}")

    _log("\nGenerating figures...")
    for variant_key, variant_folder in VARIANTS:
        for level in LEVELS:
            _log(f"  [{variant_folder.upper()}] level {level}%")
            generate_figures_for(df, variant_key, variant_folder, level)

    _log(f"\nDone — {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
