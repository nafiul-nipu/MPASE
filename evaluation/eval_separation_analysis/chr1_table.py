"""
Chr1 separation table — YZ plane, 100% of points.
Three specific comparison pairs × four methods.
Produces a CSV, LaTeX table, and a publication-quality figure.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

CHROM     = "chr1"
PLANE     = "YZ"
LEVEL     = 100
DATA_ROOT = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
XYZ_COLS  = ("middle_x", "middle_y", "middle_z")
OUT_DIR   = os.path.dirname(__file__)

# The three pairs that define the table
PAIRS = [
    ("Stability", "UNTR 12h vs 18h", "12hrs_untr", "18hrs_untr"),
    ("Temporal",  "VACV 12h vs 18h", "12hrs_vacv", "18hrs_vacv"),
    ("Condition", "UNTR vs VACV @ 18h", "18hrs_untr", "18hrs_vacv"),
]

METHODS = ["ConvexHull", "AlphaShape", "HDR", "PF"]

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

CFG_HDR = mpase.CfgHDR(n_boot=256, mass_levels=(1.00,))
CFG_PF  = mpase.CfgPF(frac_levels=(1.00,))


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_mpase():
    # load only the 4 conditions needed for the 3 pairs
    needed = {"12hrs_untr", "18hrs_untr", "12hrs_vacv", "18hrs_vacv"}
    csvs, labels = [], []
    for key in sorted(needed):
        hrs, cond = key.split("_", 1)
        p = os.path.join(DATA_ROOT, CHROM, hrs, cond,
                         f"structure_{hrs}_{cond}_gene_info.csv")
        csvs.append(p)
        labels.append(f"{CHROM}_{key}")
    return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                     cfg_hdr=CFG_HDR, cfg_pf=CFG_PF)


def get_mask(result, label):
    """Return {method: mask} for one label at 100% on YZ plane."""
    proj   = result["projections"][PLANE]
    xs, ys = proj["xs"], proj["ys"]
    pts2d  = proj["sets"][label]
    zero   = np.zeros((len(ys), len(xs)), dtype=bool)

    ch_mask = convex_hull_mask(pts2d, xs, ys)

    alpha   = heuristic_alpha(pts2d)
    as_mask, _, fell_back = alpha_shape_mask(pts2d, xs, ys, alpha=alpha)
    if fell_back:
        _log(f"  WARNING: AlphaShape fallback for {label}")

    hdr_sp = (result["shapes"].get("hdr", {})
              .get(PLANE, {}).get(LEVEL, {}).get(label))
    pf_sp  = (result["shapes"].get("point_fraction", {})
              .get(PLANE, {}).get(LEVEL, {}).get(label))

    return {
        "ConvexHull": ch_mask,
        "AlphaShape": as_mask,
        "HDR":        hdr_sp["mask"].astype(bool) if hdr_sp else zero,
        "PF":         pf_sp["mask"].astype(bool)  if pf_sp  else zero,
    }


def compute_metrics(ma, mb):
    iou       = iou_bool(ma, mb)
    meannn, _ = contour_distances(mask_to_contour(ma), mask_to_contour(mb))
    return round(iou, 4), round(meannn, 3)


# ── table & figure ────────────────────────────────────────────────────────────

def build_table(result):
    # pre-compute masks for all 4 unique labels
    labels_needed = set()
    for _, _, ka, kb in PAIRS:
        labels_needed.add(f"{CHROM}_{ka}")
        labels_needed.add(f"{CHROM}_{kb}")
    mask_cache = {lbl: get_mask(result, lbl) for lbl in labels_needed}

    rows = []
    for method in METHODS:
        row = {"Method": method}
        for cat, pair_name, ka, kb in PAIRS:
            la, lb    = f"{CHROM}_{ka}", f"{CHROM}_{kb}"
            iou, mnn  = compute_metrics(mask_cache[la][method],
                                        mask_cache[lb][method])
            row[cat]         = iou
            row[f"{cat}_mNN"] = mnn
            _log(f"  {method:12} {cat:12} IoU={iou:.4f}  meanNN={mnn:.3f}")

        row["Gap"] = round(row["Stability"] - row["Condition"], 4)
        rows.append(row)

    return pd.DataFrame(rows)


def save_csv_and_latex(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "chr1_yz_100pct_table.csv")
    df.to_csv(csv_path, index=False)
    _log(f"  Saved {csv_path}")

    # LaTeX
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Shape method comparison for " + CHROM.upper() +
        r", YZ plane, 100\% of points. "
        r"Stability = UNTR 12h vs 18h; "
        r"Temporal = VACV 12h vs 18h; "
        r"Condition = UNTR vs VACV at 18h. "
        r"Gap = Stability IoU $-$ Condition IoU; higher = better discrimination.}",
        r"\label{tab:chr1_yz_100}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Stability IoU & Temporal IoU & Condition IoU & Gap \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Method']} & {row['Stability']:.4f} & "
            f"{row['Temporal']:.4f} & {row['Condition']:.4f} & "
            f"{row['Gap']:+.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(OUT_DIR, "chr1_yz_100pct_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    _log(f"  Saved {tex_path}")


def make_figure(df):
    """
    Two-panel figure:
      Left:  grouped bar — IoU for each comparison type per method
      Right: gap bar — Stability − Condition gap per method
    Row order on left panel: Stability > Temporal > Condition (expected)
    """
    cats = ["Stability", "Temporal", "Condition"]
    x    = np.arange(len(METHODS))
    w    = 0.22

    fig, (ax_bar, ax_gap) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [2, 1]})

    # ── left: grouped bars ────────────────────────────────────────────────────
    offsets = [-w, 0, w]
    for offset, cat in zip(offsets, cats):
        vals = df[cat].values
        bars = ax_bar.bar(x + offset, vals, w,
                          label=cat, color=CAT_COLORS[cat], alpha=0.88,
                          edgecolor=[PALETTE[m] for m in METHODS], linewidth=1.8)
        for xi, v in zip(x + offset, vals):
            ax_bar.text(xi, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8,
                        color=CAT_COLORS[cat], fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(METHODS, fontsize=11)
    ax_bar.set_ylabel("IoU  (100% of points)", fontsize=11)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title(f"{CHROM.upper()}  |  YZ plane  |  100% of points\n"
                     f"Expected order: Stability > Temporal > Condition", fontsize=10)
    ax_bar.legend(fontsize=10, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)

    # annotate expected order violation in red
    for i, method in enumerate(METHODS):
        row   = df[df["Method"] == method].iloc[0]
        stab  = row["Stability"]
        temp  = row["Temporal"]
        cond  = row["Condition"]
        order_ok = stab >= temp >= cond
        if not order_ok:
            ax_bar.text(i, 0.02, "⚠", ha="center", va="bottom",
                        fontsize=14, color="red", alpha=0.7)

    # ── right: gap bar ────────────────────────────────────────────────────────
    gaps   = df["Gap"].values
    colors = [PALETTE[m] for m in METHODS]
    bars   = ax_gap.bar(x, gaps, 0.5, color=colors, alpha=0.88, edgecolor="white",
                        linewidth=1.2)

    for xi, v, m in zip(x, gaps, METHODS):
        ypos = v + 0.003 if v >= 0 else v - 0.015
        ax_gap.text(xi, ypos, f"{v:+.4f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, color=PALETTE[m], fontweight="bold")

    ax_gap.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(METHODS, fontsize=11)
    ax_gap.set_ylabel("Gap  (Stability IoU − Condition IoU)", fontsize=10)
    ax_gap.set_title("Discriminative gap\n(higher = better)", fontsize=10)
    ax_gap.grid(axis="y", alpha=0.3)

    # legend patches for method colours
    handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in METHODS]
    ax_gap.legend(handles=handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "chr1_yz_100pct_figure.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    _log(f"Chr1 YZ 100% table — {CHROM.upper()}")

    _log("Running MPASE (4 conditions)...")
    result = run_mpase()
    _log(f"MPASE done — {time.time()-t0:.1f}s")

    _log("Computing metrics...")
    df = build_table(result)

    _log("\n  === Table ===")
    _log(df[["Method", "Stability", "Temporal", "Condition", "Gap"]].to_string(index=False))

    _log("\nSaving CSV + LaTeX...")
    save_csv_and_latex(df)

    _log("Generating figure...")
    make_figure(df)

    _log(f"\nDone — {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
