"""
Baseline Comparison: Convex Hull & Alpha Shape vs MPASE (HDR 100% / PF 100%)

Fair comparison protocol:
  - All methods use the SAME aligned 3D point clouds and projection grids from MPASE
  - Baselines (ConvexHull, AlphaShape) operate on the FULL projected point set (no density filtering)
  - MPASE shapes are taken at the 100% density level (HDR-100, PF-100)
  - All four masks are rasterized onto the identical shared grid
  - Metrics (IoU, meanNN) are computed under the same conditions for all methods

The 100%-only restriction is intentional: convex hull and alpha shape do not have
a natural notion of 60/80/95% coverage levels, so comparing at 100% is the only
fair common ground.

Alpha shape uses a fast KDTree heuristic for alpha (no optimizealpha binary search).
Fallbacks to convex hull are logged explicitly so they are visible in the output.

Usage:
    source venv/bin/activate
    python -u evaluation/eval_baseline_comparison/run.py 2>&1 | tee /tmp/baseline_run.log
    # watch live in another terminal: tail -f /tmp/baseline_run.log
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(line_buffering=True)   # flush every line even when piped

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import mpase
from mpase.metrics_calculation import iou_bool, contour_distances
from baselines import (
    convex_hull_mask, alpha_shape_mask, heuristic_alpha,
    mask_to_contour, contour_to_physical,
)

warnings.filterwarnings("ignore")


def _log(msg: str):
    """Timestamped print — visible immediately even when output is redirected."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── config ────────────────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
XYZ_COLS  = ("middle_x", "middle_y", "middle_z")
CHROMS    = ["chr21", "chr7", "chr25"]
PLANES    = ["YZ", "XZ"]

# 100% only — the only level where ConvexHull/AlphaShape and HDR/PF are comparable
LEVEL   = 100
METHODS = ["ConvexHull", "AlphaShape", "HDR", "PF"]

# n_boot=256 matches all other evaluations in this project
CFG_HDR = mpase.CfgHDR(n_boot=256, mass_levels=(1.00,))
CFG_PF  = mpase.CfgPF(frac_levels=(1.00,))

# 9 comparison pairs: within-condition stability + between-condition differences
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

PALETTE = {
    "ConvexHull": "#e15759",
    "AlphaShape": "#f28e2b",
    "HDR":        "#4e79a7",
    "PF":         "#59a14f",
}


# ── Step 1: MPASE pipeline ────────────────────────────────────────────────────

def run_mpase(chrom: str) -> dict:
    """
    Load all 6 conditions and run MPASE:
      PCA+ICP alignment → shared 2D projection grids → HDR-100 and PF-100 shapes.
    All baseline methods will reuse the aligned point clouds and grids from this result.
    """
    csvs, labels = [], []
    for hrs in ["12hrs", "18hrs", "24hrs"]:
        for cond in ["untr", "vacv"]:
            p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                             f"structure_{hrs}_{cond}_gene_info.csv")
            csvs.append(p)
            labels.append(f"{chrom}_{hrs}_{cond}")
    _log(f"    Loading {len(csvs)} conditions: {labels}")
    return mpase.run(csv_list=csvs, labels=labels, xyz_cols=XYZ_COLS,
                     cfg_hdr=CFG_HDR, cfg_pf=CFG_PF)


# ── Step 2: compute one mask per (label, plane) for all four methods ──────────

def compute_all_masks(result: dict, chrom: str) -> dict:
    """
    Build masks[label][plane] = {method: bool_mask} for all 4 methods.

    ConvexHull and AlphaShape use the FULL aligned projected point set — no density
    filtering — so the comparison is fair at the 100% level.

    AlphaShape uses the KDTree heuristic for alpha (instant, no optimizealpha).
    Fallbacks to convex hull are logged with a WARNING tag so they stand out.

    Returns masks dict and a fallback log list.
    """
    # unique labels in order
    seen = {}
    for _, _, ka, kb in COMPARISONS:
        seen[f"{chrom}_{ka}"] = None
        seen[f"{chrom}_{kb}"] = None
    all_labels = list(seen.keys())

    masks    = {}
    fallback_log = []
    total    = len(all_labels) * len(PLANES)
    done     = 0

    _log(f"    Computing masks for {len(all_labels)} labels × {len(PLANES)} planes "
         f"= {total} pairs")

    for label in all_labels:
        masks[label] = {}
        for plane in PLANES:
            proj   = result["projections"][plane]
            xs, ys = proj["xs"], proj["ys"]
            pts2d  = proj["sets"][label]   # full aligned projected point set

            t0 = time.time()

            # --- baselines: full point set, no density filtering ---
            ch_mask = convex_hull_mask(pts2d, xs, ys)

            alpha_val = heuristic_alpha(pts2d)
            as_mask, alpha_used, fell_back = alpha_shape_mask(pts2d, xs, ys, alpha=alpha_val)

            if fell_back:
                msg = f"WARNING: AlphaShape fell back to ConvexHull — {label} {plane} (alpha={alpha_val:.4f})"
                _log(f"    {msg}")
                fallback_log.append(msg)

            # --- MPASE shapes at 100% level ---
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

            done += 1
            elapsed = time.time() - t0
            _log(f"    [{done:2d}/{total}] {label}  {plane}  "
                 f"alpha={alpha_used:.4f}  fallback={fell_back}  "
                 f"CH={ch_mask.sum()}  AS={as_mask.sum()}  "
                 f"HDR={hdr_mask.sum()}  PF={pf_mask.sum()}  "
                 f"[{elapsed:.2f}s]")

    return masks, fallback_log


# ── Step 3: metrics ───────────────────────────────────────────────────────────

def compute_pair_metrics(masks_a: dict, masks_b: dict) -> dict:
    """
    IoU and meanNN between condition A and B for each method.
    All four methods are evaluated identically.
    """
    out = {}
    for method in METHODS:
        ma, mb      = masks_a[method], masks_b[method]
        iou         = iou_bool(ma, mb)
        ca          = mask_to_contour(ma)
        cb          = mask_to_contour(mb)
        meannn, _   = contour_distances(ca, cb)
        out[method] = {"IoU": round(iou, 4), "meanNN": round(meannn, 3)}
    return out


def build_rows(masks: dict, chrom: str) -> list:
    """Build one row per (comparison, plane) from the precomputed mask cache."""
    rows  = []
    total = len(COMPARISONS) * len(PLANES)
    done  = 0

    for comp_name, category, key_a, key_b in COMPARISONS:
        label_a = f"{chrom}_{key_a}"
        label_b = f"{chrom}_{key_b}"
        for plane in PLANES:
            metrics = compute_pair_metrics(masks[label_a][plane], masks[label_b][plane])
            done += 1
            row = {"chrom": chrom, "comparison": comp_name,
                   "category": category, "plane": plane}
            for method in METHODS:
                row[f"{method}_IoU"]    = metrics[method]["IoU"]
                row[f"{method}_meanNN"] = metrics[method]["meanNN"]
            rows.append(row)
            _log(f"    metrics [{done:2d}/{total}] {plane}  {comp_name:25}  "
                 f"CH={metrics['ConvexHull']['IoU']:.3f}  "
                 f"AS={metrics['AlphaShape']['IoU']:.3f}  "
                 f"HDR={metrics['HDR']['IoU']:.3f}  "
                 f"PF={metrics['PF']['IoU']:.3f}")
    return rows


# ── Step 4: save tables ────────────────────────────────────────────────────────

def save_tables(df: pd.DataFrame, chrom: str, fallback_log: list):
    """Save full CSV, display CSV with IoU/meanNN format, LaTeX table, and fallback log."""
    res_dir = os.path.join(os.path.dirname(__file__), chrom, "results")
    os.makedirs(res_dir, exist_ok=True)

    # full numeric CSV
    df.to_csv(os.path.join(res_dir, "comparison_100pct.csv"), index=False)

    # readable display: "IoU/meanNN"
    display = df[["comparison", "category", "plane"]].copy()
    for method in METHODS:
        display[method] = (df[f"{method}_IoU"].map("{:.3f}".format) + "/" +
                           df[f"{method}_meanNN"].map("{:.2f}".format))
    display_path = os.path.join(res_dir, "comparison_100pct_display.csv")
    display.to_csv(display_path, index=False)
    _log(f"    Saved {display_path}")

    # LaTeX table
    tex_path = os.path.join(res_dir, "table_100pct.tex")
    _write_latex(display, tex_path, chrom)
    _log(f"    Saved {tex_path}")

    # fallback log
    if fallback_log:
        fb_path = os.path.join(res_dir, "alphashape_fallbacks.txt")
        with open(fb_path, "w") as f:
            f.write("\n".join(fallback_log))
        _log(f"    Saved {fb_path}  ({len(fallback_log)} fallback(s))")
    else:
        _log(f"    No AlphaShape fallbacks.")


def _write_latex(df: pd.DataFrame, path: str, chrom: str):
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Baseline comparison at 100\% density --- " + chrom.upper() +
        r". ConvexHull and AlphaShape operate on the full aligned projected point set. "
        r"HDR and PF are MPASE shapes at the 100\% level. "
        r"Values: IoU/meanNN (higher IoU = better; lower meanNN = better).}",
        r"\label{tab:baseline_100_" + chrom + r"}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Comparison & Category & Plane & ConvexHull & AlphaShape & HDR & PF \\",
        r"\midrule",
    ]
    prev_comp = None
    for _, row in df.iterrows():
        comp = row["comparison"] if row["comparison"] != prev_comp else ""
        prev_comp = row["comparison"]
        cat  = row["category"] if comp else ""
        lines.append(f"{_tex(comp)} & {_tex(cat)} & {row['plane']} & "
                     f"{row['ConvexHull']} & {row['AlphaShape']} & "
                     f"{row['HDR']} & {row['PF']} \\\\")
        if comp and row["plane"] == PLANES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("@", r"at")


# ── Step 5: figures ────────────────────────────────────────────────────────────

def fig_overlay(masks: dict, result: dict, chrom: str, plane: str,
                comp_key: str = "18hrs_untr", comp_key2: str = "18hrs_vacv",
                comp_label: str = "UNTR vs VACV @ 18h"):
    """
    Side-by-side: all points for condition A and B with all 4 method contours overlaid.
    Sanity-check figure — confirms baselines and MPASE shapes cover the same data.
    """
    fig_dir = os.path.join(os.path.dirname(__file__), chrom, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    label_a = f"{chrom}_{comp_key}"
    label_b = f"{chrom}_{comp_key2}"
    proj    = result["projections"][plane]
    xs, ys  = proj["xs"], proj["ys"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label, title in [
        (axes[0], label_a, comp_key.replace("hrs_", "h ").upper()),
        (axes[1], label_b, comp_key2.replace("hrs_", "h ").upper()),
    ]:
        pts = proj["sets"][label]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, c="lightgray", alpha=0.5, label="Points")
        for method in METHODS:
            contour = mask_to_contour(masks[label][plane][method])
            if contour is None:
                continue
            phys = contour_to_physical(contour, xs, ys)
            ax.plot(phys[:, 0], phys[:, 1], color=PALETTE[method],
                    linewidth=2, label=method)
        ax.set_title(f"{title} — {plane}, 100%")
        ax.set_xlabel("Axis 0"); ax.set_ylabel("Axis 1")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_aspect("equal")

    fig.suptitle(f"{comp_label} — {chrom.upper()} ({plane})")
    plt.tight_layout()
    out = os.path.join(fig_dir, f"overlay_{comp_key}_vs_{comp_key2}_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"    Saved {out}")


def fig_iou_heatmap(df: pd.DataFrame, chrom: str, plane: str):
    """Heatmap: IoU for all 9 comparisons × 4 methods."""
    fig_dir = os.path.join(os.path.dirname(__file__), chrom, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sub        = df[df["plane"] == plane]
    comp_order = [c[0] for c in COMPARISONS]
    pivot = pd.DataFrame({
        m: sub.set_index("comparison")[f"{m}_IoU"].reindex(comp_order)
        for m in METHODS
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5, cbar_kws={"label": "IoU"})
    ax.set_title(f"IoU — {chrom.upper()}, {plane}, 100%  (higher = better)")
    plt.tight_layout()
    out = os.path.join(fig_dir, f"heatmap_iou_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"    Saved {out}")


def fig_meannn_heatmap(df: pd.DataFrame, chrom: str, plane: str):
    """Heatmap: meanNN for all 9 comparisons × 4 methods (lower = better)."""
    fig_dir = os.path.join(os.path.dirname(__file__), chrom, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sub        = df[df["plane"] == plane]
    comp_order = [c[0] for c in COMPARISONS]
    pivot = pd.DataFrame({
        m: sub.set_index("comparison")[f"{m}_meanNN"].reindex(comp_order)
        for m in METHODS
    })
    # cap colour scale at 20 so small differences are visible
    vmax = min(20, pivot.values[np.isfinite(pivot.values)].max())
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=vmax, ax=ax, linewidths=0.5, cbar_kws={"label": "meanNN"})
    ax.set_title(f"meanNN — {chrom.upper()}, {plane}, 100%  (lower = better)")
    plt.tight_layout()
    out = os.path.join(fig_dir, f"heatmap_meannn_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"    Saved {out}")


def fig_sensitivity_bar(df: pd.DataFrame, chrom: str, plane: str):
    """
    Bar chart: mean IoU for Stability vs Condition per method.
    A larger gap means the method better discriminates between stable and changed structures.
    """
    fig_dir = os.path.join(os.path.dirname(__file__), chrom, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sub  = df[df["plane"] == plane]
    rows = [{"method": m, "category": cat,
             "mean_IoU": sub[sub["category"] == cat][f"{m}_IoU"].mean()}
            for m in METHODS for cat in ["Stability", "Condition"]]
    dfs  = pd.DataFrame(rows)

    x, width = np.arange(len(METHODS)), 0.35
    fig, ax  = plt.subplots(figsize=(8, 4))
    for i, (cat, color) in enumerate([("Stability", "#aec7e8"), ("Condition", "#ffbb78")]):
        vals = [dfs[(dfs["method"] == m) & (dfs["category"] == cat)]["mean_IoU"].values[0]
                for m in METHODS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=cat,
               color=color, alpha=0.85, edgecolor=[PALETTE[m] for m in METHODS], linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 0.5) * width, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.set_ylabel("Mean IoU"); ax.set_ylim(0, 1)
    ax.set_title(f"Stability vs Condition — {chrom.upper()}, {plane}, 100%\n"
                 f"(gap = sensitivity to structural change)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, f"sensitivity_bar_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"    Saved {out}")


def fig_boxplot(df: pd.DataFrame, chrom: str, plane: str):
    """Boxplot: IoU distribution per method split by comparison category."""
    fig_dir = os.path.join(os.path.dirname(__file__), chrom, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sub    = df[df["plane"] == plane]
    melted = sub.melt(id_vars=["comparison", "category"],
                      value_vars=[f"{m}_IoU" for m in METHODS],
                      var_name="method", value_name="IoU")
    melted["method"] = melted["method"].str.replace("_IoU", "")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=melted, x="category", y="IoU", hue="method",
                palette=PALETTE, ax=ax)
    ax.set_title(f"IoU distribution by category — {chrom.upper()}, {plane}, 100%")
    ax.set_ylim(0, 1); ax.legend(title="Method", fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, f"boxplot_by_category_{plane}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"    Saved {out}")


# ── orchestration ─────────────────────────────────────────────────────────────

def run_chrom(chrom: str) -> tuple:
    """Full pipeline for one chromosome. Returns (df, timing_dict)."""
    timing      = {}
    chrom_start = time.time()

    _log(f"")
    _log(f"{'='*60}")
    _log(f"  CHROMOSOME: {chrom.upper()}")
    _log(f"{'='*60}")

    # --- 1: MPASE alignment + projection + HDR/PF shapes ---
    t0 = time.time()
    _log(f"  [1/5] MPASE pipeline (n_boot={CFG_HDR.n_boot}, level=100%)...")
    result = run_mpase(chrom)
    timing["mpase_s"] = round(time.time() - t0, 1)
    _log(f"  [1/5] done — {timing['mpase_s']}s")

    # --- 2: compute all method masks (full point set, no density filtering) ---
    t0 = time.time()
    _log(f"  [2/5] Computing masks (ConvexHull + AlphaShape on full points, HDR/PF from MPASE)...")
    masks, fallback_log = compute_all_masks(result, chrom)
    timing["masks_s"] = round(time.time() - t0, 1)
    _log(f"  [2/5] done — {timing['masks_s']}s  ({len(fallback_log)} AlphaShape fallback(s))")

    # --- 3: pair metrics ---
    t0 = time.time()
    _log(f"  [3/5] Computing pair metrics ({len(COMPARISONS)} pairs × {len(PLANES)} planes)...")
    rows = build_rows(masks, chrom)
    df   = pd.DataFrame(rows)
    timing["metrics_s"] = round(time.time() - t0, 1)
    _log(f"  [3/5] done — {timing['metrics_s']}s")

    # --- 4: tables ---
    t0 = time.time()
    _log(f"  [4/5] Saving tables...")
    save_tables(df, chrom, fallback_log)
    timing["tables_s"] = round(time.time() - t0, 1)
    _log(f"  [4/5] done — {timing['tables_s']}s")

    # --- 5: figures ---
    t0 = time.time()
    _log(f"  [5/5] Generating figures...")
    for plane in PLANES:
        fig_overlay(masks, result, chrom, plane=plane,
                    comp_key="18hrs_untr", comp_key2="18hrs_vacv",
                    comp_label="UNTR vs VACV @ 18h")
        fig_iou_heatmap(df, chrom, plane)
        fig_meannn_heatmap(df, chrom, plane)
        fig_sensitivity_bar(df, chrom, plane)
        fig_boxplot(df, chrom, plane)
    timing["figures_s"] = round(time.time() - t0, 1)
    _log(f"  [5/5] done — {timing['figures_s']}s")

    timing["total_s"] = round(time.time() - chrom_start, 1)
    _log(f"")
    _log(f"  {chrom.upper()} DONE — {timing['total_s']}s  "
         f"(MPASE={timing['mpase_s']}s  masks={timing['masks_s']}s  "
         f"metrics={timing['metrics_s']}s  tables={timing['tables_s']}s  "
         f"figs={timing['figures_s']}s)")

    res_dir = os.path.join(os.path.dirname(__file__), chrom, "results")
    os.makedirs(res_dir, exist_ok=True)
    pd.DataFrame([timing]).to_csv(os.path.join(res_dir, "timing.csv"), index=False)

    return df, timing


def main():
    run_start  = time.time()
    all_dfs    = []
    all_timing = []

    _log(f"Baseline comparison (100% level only) — chromosomes: {CHROMS}")

    for chrom in CHROMS:
        df, timing = run_chrom(chrom)
        timing["chrom"] = chrom
        all_dfs.append(df)
        all_timing.append(timing)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_dir  = os.path.dirname(__file__)
    combined.to_csv(os.path.join(out_dir, "combined_all_chroms.csv"), index=False)
    _log(f"Saved combined_all_chroms.csv")

    cols = ["chrom", "mpase_s", "masks_s", "metrics_s", "tables_s", "figures_s", "total_s"]
    timing_df = pd.DataFrame(all_timing)[cols]
    timing_df.to_csv(os.path.join(out_dir, "timing_summary.csv"), index=False)

    total_s = round(time.time() - run_start, 1)
    _log(f"")
    _log(f"{'='*60}")
    _log(f"  COMPLETE — {total_s}s ({total_s/60:.1f} min)")
    _log(timing_df.to_string(index=False))
    _log(f"{'='*60}")

    # cross-chromosome sensitivity figure
    fig, axes = plt.subplots(1, len(CHROMS), figsize=(5 * len(CHROMS), 4), sharey=True)
    for ax, chrom in zip(axes, CHROMS):
        sub  = combined[(combined["chrom"] == chrom) & (combined["plane"] == "YZ")]
        rows = [{"method": m, "category": cat,
                 "mean_IoU": sub[sub["category"] == cat][f"{m}_IoU"].mean()}
                for m in METHODS for cat in ["Stability", "Condition"]]
        dfs  = pd.DataFrame(rows)
        x, width = np.arange(len(METHODS)), 0.35
        for i, (cat, color) in enumerate([("Stability", "#aec7e8"), ("Condition", "#ffbb78")]):
            vals = [dfs[(dfs["method"] == m) & (dfs["category"] == cat)]["mean_IoU"].values[0]
                    for m in METHODS]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=cat if ax == axes[0] else "",
                   color=color, alpha=0.85,
                   edgecolor=[PALETTE[m] for m in METHODS], linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
        ax.set_title(chrom.upper()); ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("Mean IoU"); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Method sensitivity: Stability vs Condition (YZ, 100% level)")
    plt.tight_layout()
    out = os.path.join(out_dir, "cross_chrom_sensitivity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"Saved {out}")


if __name__ == "__main__":
    main()
