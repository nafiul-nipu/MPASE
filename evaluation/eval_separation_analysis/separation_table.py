"""
Separation table — YZ plane, 100% of points, multiple chromosomes.
Three specific comparison pairs × four methods × N chromosomes.

Outputs per chromosome:
  results/{chrom}_table.csv
  results/{chrom}_table.tex
  figures/{chrom}_figure.png

Combined outputs:
  results/all_chroms_separation.csv
  figures/gap_heatmap.png       — Gap (Stability − Condition) per chrom × method
  figures/iou_heatmap.png       — IoU values per chrom × method × category (3-panel)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "evaluation", "eval_baseline_comparison"))

import mpase
from mpase.metrics_calculation import iou_bool, contour_distances
from baselines import (
    convex_hull_mask, alpha_shape_mask, heuristic_alpha, mask_to_contour,
)

warnings.filterwarnings("ignore")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── config ────────────────────────────────────────────────────────────────────

CHROMS    = ["chr1", "chr7", "chr21", "chr25"]
DATA_ROOT = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
OUT_DIR   = os.path.dirname(__file__)
RES_DIR   = os.path.join(OUT_DIR, "results")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
XYZ_COLS  = ("middle_x", "middle_y", "middle_z")
PLANE     = "YZ"
LEVEL     = 100
METHODS   = ["ConvexHull", "AlphaShape", "HDR", "PF"]

# Same three pairs for every chromosome
PAIRS = [
    ("Stability", "UNTR 12h vs 18h",    "12hrs_untr", "18hrs_untr"),
    ("Temporal",  "VACV 12h vs 18h",    "12hrs_vacv", "18hrs_vacv"),
    ("Condition", "UNTR vs VACV @ 18h", "18hrs_untr", "18hrs_vacv"),
]

CFG_HDR = mpase.CfgHDR(n_boot=256, mass_levels=(1.00,))
CFG_PF  = mpase.CfgPF(frac_levels=(1.00,))

PALETTE = {
    "ConvexHull": "#e15759",
    "AlphaShape": "#f28e2b",
    "HDR":        "#4e79a7",
    "PF":         "#59a14f",
}
CAT_COLORS = {
    "Stability": "#5778a4",
    "Temporal":  "#e49444",
    "Condition": "#d1615d",
}


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_mpase(chrom: str) -> dict:
    needed = {"12hrs_untr", "18hrs_untr", "12hrs_vacv", "18hrs_vacv"}
    csvs, labels = [], []
    for key in sorted(needed):
        hrs, cond = key.split("_", 1)
        p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                         f"structure_{hrs}_{cond}_gene_info.csv")
        csvs.append(p)
        labels.append(f"{chrom}_{key}")
    return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                     cfg_hdr=CFG_HDR, cfg_pf=CFG_PF)


def get_masks(result, chrom):
    """Return masks[label] = {method: mask} for all needed labels."""
    needed = set()
    for _, _, ka, kb in PAIRS:
        needed.add(f"{chrom}_{ka}")
        needed.add(f"{chrom}_{kb}")

    cache = {}
    proj   = result["projections"][PLANE]
    xs, ys = proj["xs"], proj["ys"]
    zero   = np.zeros((len(ys), len(xs)), dtype=bool)

    for label in needed:
        pts2d = proj["sets"][label]
        ch    = convex_hull_mask(pts2d, xs, ys)
        alpha = heuristic_alpha(pts2d)
        asmask, _, fb = alpha_shape_mask(pts2d, xs, ys, alpha=alpha)
        if fb:
            _log(f"  WARNING: AlphaShape fallback for {label}")

        hdr_sp = result["shapes"].get("hdr",{}).get(PLANE,{}).get(LEVEL,{}).get(label)
        pf_sp  = result["shapes"].get("point_fraction",{}).get(PLANE,{}).get(LEVEL,{}).get(label)

        cache[label] = {
            "ConvexHull": ch,
            "AlphaShape": asmask,
            "HDR":  hdr_sp["mask"].astype(bool) if hdr_sp else zero,
            "PF":   pf_sp["mask"].astype(bool)  if pf_sp  else zero,
        }
    return cache


def build_table(masks, chrom):
    rows = []
    for method in METHODS:
        row = {"Method": method}
        for cat, _, ka, kb in PAIRS:
            la, lb    = f"{chrom}_{ka}", f"{chrom}_{kb}"
            iou       = iou_bool(masks[la][method], masks[lb][method])
            mnn, _    = contour_distances(mask_to_contour(masks[la][method]),
                                          mask_to_contour(masks[lb][method]))
            row[cat]          = round(iou, 4)
            row[f"{cat}_mNN"] = round(mnn, 3)
        row["Gap"] = round(row["Stability"] - row["Condition"], 4)
        rows.append(row)
        _log(f"  {method:12}  S={row['Stability']:.3f}  T={row['Temporal']:.3f}  "
             f"C={row['Condition']:.3f}  gap={row['Gap']:+.3f}")
    return pd.DataFrame(rows)


# ── save ──────────────────────────────────────────────────────────────────────

def save_chrom(df, chrom):
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, f"{chrom}_table.csv"), index=False)

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Separation analysis: " + chrom.upper() +
        r", YZ plane, 100\% of points. Gap = Stability\,IoU $-$ Condition\,IoU.}",
        r"\label{tab:sep_" + chrom + r"}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Method & Stability & Temporal & Condition & Gap \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['Method']} & {r['Stability']:.4f} & {r['Temporal']:.4f} "
                     f"& {r['Condition']:.4f} & {r['Gap']:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RES_DIR, f"{chrom}_table.tex"), "w") as f:
        f.write("\n".join(lines))


# ── per-chromosome figure ──────────────────────────────────────────────────────

def fig_chrom(df, chrom):
    os.makedirs(FIG_DIR, exist_ok=True)
    cats = ["Stability", "Temporal", "Condition"]
    x, w = np.arange(len(METHODS)), 0.22

    fig, (ax_bar, ax_gap) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [2, 1]})

    for offset, cat in zip([-w, 0, w], cats):
        vals = df[cat].values
        ax_bar.bar(x + offset, vals, w, label=cat,
                   color=CAT_COLORS[cat], alpha=0.88,
                   edgecolor=[PALETTE[m] for m in METHODS], linewidth=1.8)
        for xi, v in zip(x + offset, vals):
            ax_bar.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=CAT_COLORS[cat], fontweight="bold")

    ax_bar.set_xticks(x); ax_bar.set_xticklabels(METHODS, fontsize=11)
    ax_bar.set_ylabel("IoU  (100% of points)", fontsize=11)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title(f"{chrom.upper()}  |  YZ plane  |  100% of points\n"
                     f"Expected: Stability > Temporal > Condition", fontsize=10)
    ax_bar.legend(fontsize=10); ax_bar.grid(axis="y", alpha=0.3)

    # warn if order violated
    for i, method in enumerate(METHODS):
        r = df[df["Method"] == method].iloc[0]
        if not (r["Stability"] >= r["Temporal"] >= r["Condition"]):
            ax_bar.text(i, 0.02, "⚠", ha="center", fontsize=14, color="red", alpha=0.7)

    gaps = df["Gap"].values
    ax_gap.bar(x, gaps, 0.5, color=[PALETTE[m] for m in METHODS],
               alpha=0.88, edgecolor="white", linewidth=1.2)
    for xi, v, m in zip(x, gaps, METHODS):
        ypos = v + 0.003 if v >= 0 else v - 0.015
        ax_gap.text(xi, ypos, f"{v:+.4f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, color=PALETTE[m], fontweight="bold")
    ax_gap.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax_gap.set_xticks(x); ax_gap.set_xticklabels(METHODS, fontsize=11)
    ax_gap.set_ylabel("Gap  (Stability − Condition)", fontsize=10)
    ax_gap.set_title("Discriminative gap\n(higher = better)", fontsize=10)
    ax_gap.legend(handles=[mpatches.Patch(color=PALETTE[m], label=m) for m in METHODS],
                  fontsize=9, loc="lower right")
    ax_gap.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{chrom}_figure.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── combined heatmaps ──────────────────────────────────────────────────────────

def fig_gap_heatmap(all_df):
    """
    Heatmap: rows = chromosomes, cols = methods, values = Gap.
    Colour = how well the method discriminates (higher = better).
    """
    pivot = all_df.pivot(index="chrom", columns="Method", values="Gap")[METHODS]
    pivot.index = [c.upper() for c in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdYlGn",
                center=0, vmin=-0.05, vmax=0.25,
                ax=ax, linewidths=0.8,
                cbar_kws={"label": "Gap  (Stability IoU − Condition IoU)"})
    ax.set_title("Discriminative gap per chromosome and method\n"
                 "(YZ plane, 100% of points — higher = better discrimination)",
                 fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "gap_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_iou_heatmaps(all_df):
    """
    Three side-by-side heatmaps (one per category): rows = chrom, cols = method.
    Shows absolute IoU so you can see both level and separation at a glance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (cat, pair_name) in zip(axes, [(p[0], p[1]) for p in PAIRS]):
        pivot = all_df.pivot(index="chrom", columns="Method",
                             values=cat)[METHODS]
        pivot.index = [c.upper() for c in pivot.index]

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                    vmin=0.4, vmax=1.0, ax=ax, linewidths=0.8,
                    cbar_kws={"label": "IoU"})
        ax.set_title(f"{cat}\n({pair_name})", fontsize=9,
                     color=CAT_COLORS[cat], fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")

    fig.suptitle("IoU by category — YZ plane, 100% of points\n"
                 "Expected: Stability > Temporal > Condition within each method",
                 fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "iou_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    t_total  = time.time()
    all_rows = []

    for chrom in CHROMS:
        _log(f"\n{'='*50}")
        _log(f"  {chrom.upper()}")
        _log(f"{'='*50}")

        t0     = time.time()
        result = run_mpase(chrom)
        _log(f"  MPASE done — {time.time()-t0:.1f}s")

        masks = get_masks(result, chrom)
        df    = build_table(masks, chrom)

        save_chrom(df, chrom)
        fig_chrom(df, chrom)

        for _, row in df.iterrows():
            all_rows.append({"chrom": chrom, **row.to_dict()})

    # combined outputs
    _log(f"\n{'='*50}")
    _log("  Combined outputs")
    _log(f"{'='*50}")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(RES_DIR, "all_chroms_separation.csv"), index=False)
    _log("  Saved all_chroms_separation.csv")

    fig_gap_heatmap(all_df)
    fig_iou_heatmaps(all_df)

    # print summary table
    summary = all_df.pivot_table(index="chrom", columns="Method", values="Gap")[METHODS]
    _log(f"\n  === Gap summary (Stability − Condition IoU) ===")
    _log(summary.to_string())

    _log(f"\nDone — {time.time()-t_total:.1f}s total")


if __name__ == "__main__":
    main()
