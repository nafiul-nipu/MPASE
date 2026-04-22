"""
Separation-Based Evaluation
============================
Answers the question: can each shape method *discriminate* between
structurally similar conditions and structurally different ones?

Protocol
--------
- One chromosome (chr21 by default).
- All methods evaluated at the 100% level on MPASE-aligned full point sets,
  identical to the baseline comparison protocol.
- Comparison pairs are grouped into three categories:
    Stability  — same condition, adjacent/non-adjacent timepoints (UNTR only)
    Temporal   — same condition across time (VACV only)
    Condition  — UNTR vs VACV at the same timepoint
- For each method and category, compute mean IoU.
- Separation gap = Stability_mean_IoU − Condition_mean_IoU.
  A larger gap means the method assigns *higher* similarity to pairs that are
  genuinely similar and *lower* similarity to pairs that differ, i.e. better
  discriminative power.
- Outputs:
    results/comparison_matrix.csv       full IoU/meanNN per pair × method
    results/separation_summary.csv      mean IoU + gap per method × category
    results/separation_table.tex        LaTeX table of the summary
    figures/separation_bar_{plane}.png  grouped bar: Stability vs Condition IoU
    figures/separation_gap_{plane}.png  gap bar per method
    figures/overlay_stability_{plane}.png  contour overlay for a stability pair
    figures/overlay_condition_{plane}.png  contour overlay for a condition pair
    figures/iou_scatter_{plane}.png     scatter: Stability IoU vs Condition IoU per method

Usage:
    source venv/bin/activate
    python -u evaluation/eval_separation_analysis/run.py 2>&1 | tee /tmp/sep_run.log
"""

import os, sys, warnings, time
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
    convex_hull_mask, alpha_shape_mask, heuristic_alpha,
    mask_to_contour, contour_to_physical,
)

warnings.filterwarnings("ignore")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── config ────────────────────────────────────────────────────────────────────

CHROM     = "chr21"
DATA_ROOT = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
OUT_DIR   = os.path.dirname(__file__)
RES_DIR   = os.path.join(OUT_DIR, "results")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
XYZ_COLS  = ("middle_x", "middle_y", "middle_z")
PLANES    = ["YZ", "XZ"]
LEVEL     = 100
METHODS   = ["ConvexHull", "AlphaShape", "HDR", "PF"]

CFG_HDR = mpase.CfgHDR(n_boot=256, mass_levels=(1.00,))
CFG_PF  = mpase.CfgPF(frac_levels=(1.00,))

# Comparisons with explicit categories for separation analysis
# Stability = same condition, different timepoints (UNTR)
# Temporal  = same condition, different timepoints (VACV)
# Condition = same timepoint, different condition
COMPARISONS = [
    ("UNTR 12h vs 18h",    "Stability",   "12hrs_untr", "18hrs_untr"),
    ("UNTR 12h vs 24h",    "Stability",   "12hrs_untr", "24hrs_untr"),
    ("UNTR 18h vs 24h",    "Stability",   "18hrs_untr", "24hrs_untr"),
    ("VACV 12h vs 18h",    "Temporal",    "12hrs_vacv", "18hrs_vacv"),
    ("VACV 12h vs 24h",    "Temporal",    "12hrs_vacv", "24hrs_vacv"),
    ("VACV 18h vs 24h",    "Temporal",    "18hrs_vacv", "24hrs_vacv"),
    ("UNTR vs VACV @ 12h", "Condition",   "12hrs_untr", "12hrs_vacv"),
    ("UNTR vs VACV @ 18h", "Condition",   "18hrs_untr", "18hrs_vacv"),
    ("UNTR vs VACV @ 24h", "Condition",   "24hrs_untr", "24hrs_vacv"),
]

# representative pairs for overlay figures
OVERLAY_STABILITY = ("UNTR 12h vs 18h", "12hrs_untr", "18hrs_untr")
OVERLAY_CONDITION = ("UNTR vs VACV @ 18h", "18hrs_untr", "18hrs_vacv")

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

def run_mpase() -> dict:
    csvs, labels = [], []
    for hrs in ["12hrs", "18hrs", "24hrs"]:
        for cond in ["untr", "vacv"]:
            p = os.path.join(DATA_ROOT, CHROM, hrs, cond,
                             f"structure_{hrs}_{cond}_gene_info.csv")
            csvs.append(p)
            labels.append(f"{CHROM}_{hrs}_{cond}")
    return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                     cfg_hdr=CFG_HDR, cfg_pf=CFG_PF)


def compute_masks(result: dict) -> dict:
    """
    masks[label][plane] = {method: bool_mask}
    Baselines use the full aligned projected point set.
    """
    seen = {}
    for _, _, ka, kb in COMPARISONS:
        seen[f"{CHROM}_{ka}"] = None
        seen[f"{CHROM}_{kb}"] = None
    all_labels = list(seen.keys())

    masks = {}
    for label in all_labels:
        masks[label] = {}
        for plane in PLANES:
            proj   = result["projections"][plane]
            xs, ys = proj["xs"], proj["ys"]
            pts2d  = proj["sets"][label]

            ch_mask = convex_hull_mask(pts2d, xs, ys)
            alpha   = heuristic_alpha(pts2d)
            as_mask, _, fell_back = alpha_shape_mask(pts2d, xs, ys, alpha=alpha)
            if fell_back:
                _log(f"  WARNING: AlphaShape fallback — {label} {plane}")

            hdr_sp = (result["shapes"]
                      .get("hdr", {}).get(plane, {}).get(LEVEL, {}).get(label))
            pf_sp  = (result["shapes"]
                      .get("point_fraction", {}).get(plane, {}).get(LEVEL, {}).get(label))
            zero     = np.zeros((len(ys), len(xs)), dtype=bool)
            hdr_mask = hdr_sp["mask"].astype(bool) if hdr_sp is not None else zero
            pf_mask  = pf_sp["mask"].astype(bool)  if pf_sp  is not None else zero

            masks[label][plane] = {
                "ConvexHull": ch_mask,
                "AlphaShape": as_mask,
                "HDR":        hdr_mask,
                "PF":         pf_mask,
            }
            _log(f"  {label}  {plane}  alpha={alpha:.2f}  "
                 f"CH={ch_mask.sum()}  AS={as_mask.sum()}  "
                 f"HDR={hdr_mask.sum()}  PF={pf_mask.sum()}")
    return masks


def build_comparison_matrix(masks: dict) -> pd.DataFrame:
    """Full IoU + meanNN matrix — one row per (comparison, plane, method)."""
    rows = []
    for comp_name, category, key_a, key_b in COMPARISONS:
        la = f"{CHROM}_{key_a}"
        lb = f"{CHROM}_{key_b}"
        for plane in PLANES:
            for method in METHODS:
                ma = masks[la][plane][method]
                mb = masks[lb][plane][method]
                iou       = iou_bool(ma, mb)
                ca        = mask_to_contour(ma)
                cb        = mask_to_contour(mb)
                meannn, _ = contour_distances(ca, cb)
                rows.append({
                    "comparison": comp_name, "category": category,
                    "plane": plane, "method": method,
                    "IoU": round(iou, 4), "meanNN": round(meannn, 3),
                })
    return pd.DataFrame(rows)


def build_separation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method × plane × category: mean IoU.
    Also compute gap = Stability_mean − Condition_mean per method × plane.
    """
    rows = []
    for plane in PLANES:
        sub = df[df["plane"] == plane]
        for method in METHODS:
            msub = sub[sub["method"] == method]
            cat_means = msub.groupby("category")["IoU"].mean().to_dict()
            stab  = cat_means.get("Stability", np.nan)
            temp  = cat_means.get("Temporal",  np.nan)
            cond  = cat_means.get("Condition", np.nan)
            # gap: how much higher is within-condition IoU vs between-condition
            gap_stab_cond = round(stab - cond, 4) if not np.isnan(stab + cond) else np.nan
            gap_temp_cond = round(temp - cond, 4) if not np.isnan(temp + cond) else np.nan
            rows.append({
                "plane": plane, "method": method,
                "Stability_IoU": round(stab, 4),
                "Temporal_IoU":  round(temp, 4),
                "Condition_IoU": round(cond, 4),
                "Gap_Stab_Cond": gap_stab_cond,
                "Gap_Temp_Cond": gap_temp_cond,
            })
    return pd.DataFrame(rows)


# ── save ──────────────────────────────────────────────────────────────────────

def save_results(matrix_df: pd.DataFrame, sep_df: pd.DataFrame):
    os.makedirs(RES_DIR, exist_ok=True)

    matrix_df.to_csv(os.path.join(RES_DIR, "comparison_matrix.csv"), index=False)
    _log(f"  Saved comparison_matrix.csv")

    sep_df.to_csv(os.path.join(RES_DIR, "separation_summary.csv"), index=False)
    _log(f"  Saved separation_summary.csv")

    _write_latex(sep_df)
    _log(f"  Saved separation_table.tex")


def _write_latex(sep_df: pd.DataFrame):
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Separation analysis for " + CHROM.upper() +
        r" at 100\% density. "
        r"Mean IoU per method and category. "
        r"Gap = Stability\,IoU $-$ Condition\,IoU; "
        r"a larger gap indicates better discriminative power.}",
        r"\label{tab:separation_" + CHROM + r"}",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Plane & Method & Stability & Temporal & Condition & Gap (S$-$C) & Gap (T$-$C) \\",
        r"\midrule",
    ]
    prev_plane = None
    for _, row in sep_df.iterrows():
        plane = row["plane"] if row["plane"] != prev_plane else ""
        prev_plane = row["plane"]
        lines.append(
            f"{plane} & {row['method']} & "
            f"{row['Stability_IoU']:.3f} & {row['Temporal_IoU']:.3f} & "
            f"{row['Condition_IoU']:.3f} & {row['Gap_Stab_Cond']:.3f} & "
            f"{row['Gap_Temp_Cond']:.3f} \\\\"
        )
        if plane and row["method"] == METHODS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RES_DIR, "separation_table.tex"), "w") as f:
        f.write("\n".join(lines))


# ── figures ───────────────────────────────────────────────────────────────────

def fig_separation_bar(sep_df: pd.DataFrame, plane: str):
    """
    Grouped bar: Stability, Temporal, Condition mean IoU for each method.
    Makes it immediately obvious which method separates the categories.
    """
    sub = sep_df[sep_df["plane"] == plane]

    categories = ["Stability", "Temporal", "Condition"]
    x     = np.arange(len(METHODS))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, cat in zip(offsets, categories):
        vals = [sub[sub["method"] == m][f"{cat}_IoU"].values[0] for m in METHODS]
        bars = ax.bar(x + offset, vals, width, label=cat,
                      color=CAT_COLORS[cat], alpha=0.85,
                      edgecolor=[PALETTE[m] for m in METHODS], linewidth=1.8)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=CAT_COLORS[cat], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Mean IoU", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mean IoU by category — {CHROM.upper()}, {plane}, 100%\n"
                 f"Stability = within-condition timepoints   "
                 f"Condition = UNTR vs VACV", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"separation_bar_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_separation_gap(sep_df: pd.DataFrame, plane: str):
    """
    Bar chart of gap = Stability − Condition and Temporal − Condition per method.
    Positive gap = method scores similar pairs higher than different pairs.
    """
    sub  = sep_df[sep_df["plane"] == plane]
    x    = np.arange(len(METHODS))
    w    = 0.3

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (col, label, color) in enumerate([
        ("Gap_Stab_Cond", "Stability − Condition", "#5778a4"),
        ("Gap_Temp_Cond", "Temporal − Condition",  "#e49444"),
    ]):
        vals = [sub[sub["method"] == m][col].values[0] for m in METHODS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=label,
                      color=color, alpha=0.85,
                      edgecolor=[PALETTE[m] for m in METHODS], linewidth=1.8)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, max(v, 0) + 0.005, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("IoU gap  (higher = better discrimination)", fontsize=10)
    ax.set_title(f"Discriminative gap — {CHROM.upper()}, {plane}, 100%\n"
                 f"Gap > 0 means the method assigns higher similarity to within-condition pairs",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"separation_gap_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_iou_scatter(matrix_df: pd.DataFrame, plane: str):
    """
    Scatter: each point is one comparison pair. X = IoU for Stability/Temporal,
    Y = IoU for Condition. Points above the diagonal mean poor discrimination.
    One colour per method, shape = Stability vs Temporal.
    """
    sub    = matrix_df[matrix_df["plane"] == plane]
    stab   = sub[sub["category"].isin(["Stability", "Temporal"])]
    cond   = sub[sub["category"] == "Condition"]

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4), sharey=True)
    for ax, method in zip(axes, METHODS):
        s_vals = stab[stab["method"] == method]["IoU"].values
        c_vals = cond[cond["method"] == method]["IoU"].values
        s_cats = stab[stab["method"] == method]["category"].values

        # one point per stability/temporal pair vs mean condition IoU (for visual clarity)
        mean_cond = c_vals.mean()
        for v, cat in zip(s_vals, s_cats):
            marker = "o" if cat == "Stability" else "^"
            ax.scatter(v, mean_cond, color=PALETTE[method], marker=marker,
                       s=60, alpha=0.8, zorder=3)

        # reference line: IoU_within == IoU_between (no discrimination)
        lo, hi = 0, 1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)

        # gap annotation
        mean_stab = s_vals.mean()
        ax.annotate(f"gap={mean_stab - mean_cond:+.3f}",
                    xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=8, color=PALETTE[method], fontweight="bold")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Within-condition IoU", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Condition IoU (mean)", fontsize=9)
        ax.set_title(method, color=PALETTE[method], fontsize=10, fontweight="bold")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

    legend_elems = [
        plt.scatter([], [], marker="o", color="gray", label="Stability"),
        plt.scatter([], [], marker="^", color="gray", label="Temporal"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Discrimination scatter — {CHROM.upper()}, {plane}, 100%\n"
                 f"Points below diagonal = method scores within-condition pairs higher",
                 fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"iou_scatter_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_overlay(masks: dict, result: dict, plane: str,
                pair_label: str, key_a: str, key_b: str, tag: str):
    """
    Side-by-side contour overlay for one pair.
    Used to visually illustrate that CH stays similar while HDR/PF varies.
    """
    la    = f"{CHROM}_{key_a}"
    lb    = f"{CHROM}_{key_b}"
    proj  = result["projections"][plane]
    xs, ys = proj["xs"], proj["ys"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label, title in [
        (axes[0], la, key_a.replace("hrs_", "h ").upper()),
        (axes[1], lb, key_b.replace("hrs_", "h ").upper()),
    ]:
        pts = proj["sets"][label]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c="lightgray", alpha=0.45, zorder=1)
        for method in METHODS:
            contour = mask_to_contour(masks[label][plane][method])
            if contour is None:
                continue
            phys = contour_to_physical(contour, xs, ys)
            ax.plot(phys[:, 0], phys[:, 1], color=PALETTE[method],
                    linewidth=2, label=method, zorder=2)
        ax.set_title(f"{title}", fontsize=11)
        ax.set_xlabel("Axis 0"); ax.set_ylabel("Axis 1")
        ax.legend(fontsize=8, loc="upper right"); ax.set_aspect("equal")

    fig.suptitle(f"{pair_label} — {CHROM.upper()}, {plane}, 100%", fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"overlay_{tag}_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_iou_heatmap_per_method(matrix_df: pd.DataFrame, plane: str):
    """
    Four heatmaps (one per method): comparison × IoU, sorted by category.
    Makes it easy to see which method assigns meaningfully different scores.
    """
    sub = matrix_df[(matrix_df["plane"] == plane)]
    comp_order = [c[0] for c in COMPARISONS]

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.5 * len(METHODS), 5), sharey=True)
    for ax, method in zip(axes, METHODS):
        msub   = sub[sub["method"] == method].set_index("comparison")["IoU"]
        pivot  = msub.reindex(comp_order)
        cats   = [c[1] for c in COMPARISONS]  # category per comparison in order
        colors = [CAT_COLORS[c] for c in cats]

        im = ax.imshow(pivot.values.reshape(-1, 1), cmap="RdYlGn",
                       vmin=0, vmax=1, aspect="auto")
        for i, (v, cat) in enumerate(zip(pivot.values, cats)):
            ax.text(0, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v < 0.4 or v > 0.75 else "black")
        ax.set_yticks(range(len(comp_order)))
        if ax == axes[0]:
            ax.set_yticklabels(comp_order, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_title(method, color=PALETTE[method], fontsize=10, fontweight="bold")

        # category colour strip on the right
        for i, cat in enumerate(cats):
            ax.add_patch(plt.Rectangle((0.5, i - 0.5), 0.15, 1,
                                       color=CAT_COLORS[cat], alpha=0.7,
                                       transform=ax.transData, clip_on=False))

    # colour scale
    fig.colorbar(im, ax=axes[-1], fraction=0.04, pad=0.04, label="IoU")

    # legend for category strip
    patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"IoU per comparison — {CHROM.upper()}, {plane}, 100%", fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"iou_heatmap_per_method_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    t_start = time.time()
    _log(f"Separation analysis — {CHROM.upper()}")

    # 1. MPASE
    _log("[1/5] Running MPASE...")
    t0     = time.time()
    result = run_mpase()
    _log(f"[1/5] done — {time.time()-t0:.1f}s")

    # 2. masks
    _log("[2/5] Computing masks...")
    t0    = time.time()
    masks = compute_masks(result)
    _log(f"[2/5] done — {time.time()-t0:.1f}s")

    # 3. metrics
    _log("[3/5] Building comparison matrix and separation summary...")
    t0        = time.time()
    matrix_df = build_comparison_matrix(masks)
    sep_df    = build_separation_summary(matrix_df)
    _log(f"[3/5] done — {time.time()-t0:.1f}s")

    _log("\n  === Separation summary ===")
    _log(sep_df.to_string(index=False))
    _log("")

    # 4. save
    _log("[4/5] Saving results...")
    save_results(matrix_df, sep_df)

    # 5. figures
    _log("[5/5] Generating figures...")
    for plane in PLANES:
        fig_separation_bar(sep_df, plane)
        fig_separation_gap(sep_df, plane)
        fig_iou_scatter(matrix_df, plane)
        fig_iou_heatmap_per_method(matrix_df, plane)
        # overlay: one stability pair, one condition pair
        fig_overlay(masks, result, plane,
                    OVERLAY_STABILITY[0], OVERLAY_STABILITY[1], OVERLAY_STABILITY[2],
                    tag="stability")
        fig_overlay(masks, result, plane,
                    OVERLAY_CONDITION[0], OVERLAY_CONDITION[1], OVERLAY_CONDITION[2],
                    tag="condition")

    _log(f"\nDone — {time.time()-t_start:.1f}s total")
    _log(f"Results : {RES_DIR}")
    _log(f"Figures : {FIG_DIR}")


if __name__ == "__main__":
    main()
