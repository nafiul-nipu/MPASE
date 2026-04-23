"""
Synthetic Ground Truth Evaluation — routes through mpase.run()
===============================================================
For each synthetic shape:
  1. Define a 2D probability density f(x, y) — the "true" shape
  2. Extend to 3D: f_3d(x,y,z) = f(x,y) * N(z; 0, Z_STD)
  3. Sample TWO replicates of N_PTS 3D points → feed to mpase.run()
  4. MPASE aligns, projects to XY plane, and produces HDR + PF masks
  5. Apply ConvexHull and AlphaShape to MPASE's projected 2D points
  6. Estimate the TRUE density in MPASE coordinate space (large GT sample)
  7. Threshold GT density at each level → ground truth mask
  8. Compute IoU between each method's mask and the GT mask

Five shapes — each tests a different property:
  Ellipse    — compact blob (baseline)
  Two-lobe   — two separated clusters (ConvexHull fills the gap)
  Crescent   — non-convex (ConvexHull fills the concavity)
  Asymmetric — large + small lobe (unequal density)
  Ring       — donut / annular (ConvexHull fills the hole)

Usage:
    source venv/bin/activate
    python -u evaluation/eval_synthetic/run.py 2>&1 | tee /tmp/synth_run.log
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "evaluation", "eval_baseline_comparison"))

import mpase
from baselines import convex_hull_mask, alpha_shape_mask, heuristic_alpha

warnings.filterwarnings("ignore")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── config ────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(__file__)
RES_DIR = os.path.join(OUT_DIR, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")

LEVELS   = [95, 96, 97, 98, 99, 100]
N_PTS    = 500      # points per replicate
N_SEEDS  = 1        # one run per shape
N_BOOT   = 256      # bootstrap iterations for HDR
Z_STD    = 0.10     # z-slab thickness — thin so XY projection dominates
PLANE    = "XY"     # MPASE projection plane
METHODS  = ["ConvexHull", "AlphaShape", "HDR", "PF"]

PALETTE = {
    "ConvexHull": "#e15759",
    "AlphaShape": "#f28e2b",
    "HDR":        "#4e79a7",
    "PF":         "#59a14f",
}

CFG_HDR = mpase.CfgHDR(
    n_boot=N_BOOT,
    mass_levels=tuple(lv / 100.0 for lv in LEVELS),
    sigma_px=3.0,            # stronger smoothing for sparse N=500 point clouds
    density_floor_frac=0.01, # trim faint noise blobs before thresholding
)
CFG_PF = mpase.CfgPF(
    frac_levels=tuple(lv / 100.0 for lv in LEVELS),
    disk_px=3,               # larger rasterization disks for smoother fill
    morph=mpase.CfgMorph(
        closing=3,           # merge nearby point blobs
        opening=1,
        keep_largest=False,  # preserve two-lobe and ring topology
        fill_holes=False,    # preserve ring's inner hole
    ),
)


# ── 2D density functions ──────────────────────────────────────────────────────

def _gauss2d(x, y, cx, cy, sx, sy, rho=0.0):
    zx = (x - cx) / sx
    zy = (y - cy) / sy
    return np.exp(-0.5 / (1 - rho**2) * (zx**2 - 2*rho*zx*zy + zy**2))


def shape_ellipse(x, y):
    return _gauss2d(x, y, 0, 0, 0.45, 0.22)


def shape_two_lobe(x, y):
    return (0.5 * _gauss2d(x, y, -0.45, 0, 0.20, 0.20) +
            0.5 * _gauss2d(x, y,  0.45, 0, 0.20, 0.20))


def shape_crescent(x, y):
    r       = np.sqrt(x**2 + y**2)
    ring    = np.exp(-((r - 0.55)**2) / (2 * 0.06**2))
    suppress = np.exp(-((x + 0.0)**2 + (y + 0.55)**2) / (2 * 0.18**2))
    return np.maximum(ring - 1.8 * suppress, 0)


def shape_asymmetric(x, y):
    large = 0.80 * _gauss2d(x, y, -0.25,  0.1, 0.30, 0.22)
    small = 0.20 * _gauss2d(x, y,  0.50, -0.3, 0.13, 0.13)
    return large + small


def shape_ring(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-((r - 0.50)**2) / (2 * 0.10**2))


SHAPES = {
    "Ellipse":    shape_ellipse,
    "Two-lobe":   shape_two_lobe,
    "Crescent":   shape_crescent,
    "Asymmetric": shape_asymmetric,
    "Ring":       shape_ring,
}


# ── sampling ──────────────────────────────────────────────────────────────────

def _make_2d_grid(lim=1.5, n=300):
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    return xs, ys


def sample_3d(density_fn, n, rng):
    """Sample n 3D points: (x,y) from density_fn, z ~ N(0, Z_STD)."""
    xs, ys = _make_2d_grid()
    xx, yy = np.meshgrid(xs, ys)
    d = density_fn(xx, yy)
    d = np.maximum(d, 0)
    d /= d.sum()

    flat = d.ravel()
    idx  = rng.choice(len(flat), size=n, p=flat)
    row, col = np.unravel_index(idx, d.shape)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    x  = xs[col] + rng.uniform(-dx/2, dx/2, n)
    y  = ys[row] + rng.uniform(-dy/2, dy/2, n)
    z  = rng.normal(0, Z_STD, n)
    return np.column_stack([x, y, z]).astype(np.float32)


# ── ground truth in MPASE coordinate space ────────────────────────────────────

def _mpase_normalization(pts_rep0, pts_rep1):
    """
    Replicate MPASE's centering + scale normalization.
    MPASE centers each set by its own mean, then normalises by the combined range.
    For two replicates from the same distribution the ICP rotation is ~identity.
    """
    c0 = pts_rep0 - pts_rep0.mean(0)
    c1 = pts_rep1 - pts_rep1.mean(0)
    combined = np.vstack([c0, c1])
    scale = float((combined.max(0) - combined.min(0)).max() + 1e-8)
    return pts_rep0.mean(0).astype(float), scale


def gt_density_on_mpase_grid(density_fn, xs, ys, center0, scale):
    """
    Compute the EXACT true density in MPASE coordinate space analytically.

    MPASE transforms raw points as:
        x_mpase = (x_raw - center0[0]) / scale
        y_mpase = (y_raw - center0[1]) / scale

    So the inverse is:
        x_raw = x_mpase * scale + center0[0]
        y_raw = y_mpase * scale + center0[1]

    We evaluate density_fn at those raw coordinates — no KDE noise,
    preserves ring holes, crescent openings, etc.
    """
    xx, yy  = np.meshgrid(xs, ys)
    x_raw   = xx * scale + center0[0]
    y_raw   = yy * scale + center0[1]
    dens    = density_fn(x_raw, y_raw)
    dens    = np.maximum(dens, 0.0)
    total   = dens.sum()
    if total > 0:
        dens /= total
    return dens


def true_mask(density_grid, level):
    """Smallest region containing `level`% of probability mass."""
    flat     = density_grid.ravel()
    sorted_d = np.sort(flat)[::-1]
    cumsum   = np.cumsum(sorted_d)
    cumsum  /= cumsum[-1]
    idx      = np.searchsorted(cumsum, level / 100.0)
    thresh   = sorted_d[min(idx, len(sorted_d) - 1)]
    return density_grid >= thresh


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


# ── per-shape evaluation ──────────────────────────────────────────────────────

def evaluate_shape(shape_name, density_fn):
    """Run all methods for one shape; return (rows, vis_data)."""
    rows = []

    for seed in range(N_SEEDS):
        rng       = np.random.default_rng(seed)
        pts_rep0  = sample_3d(density_fn, N_PTS, rng)
        pts_rep1  = sample_3d(density_fn, N_PTS, rng)

        labels = [f"{shape_name}_rep0", f"{shape_name}_rep1"]

        result = mpase.run(
            points_list=[pts_rep0, pts_rep1],
            labels=labels,
            cfg_hdr=CFG_HDR,
            cfg_pf=CFG_PF,
            planes=(PLANE,),
        )

        proj       = result["projections"][PLANE]
        xs, ys     = proj["xs"], proj["ys"]
        lab        = labels[0]
        pts2d      = proj["sets"][lab]   # aligned + projected 2D points for rep0

        # GT density in MPASE coordinate space (analytical — no KDE noise)
        center0, scale = _mpase_normalization(pts_rep0, pts_rep1)
        dens_grid      = gt_density_on_mpase_grid(density_fn, xs, ys, center0, scale)
        gt_masks       = {lv: true_mask(dens_grid, lv) for lv in LEVELS}

        # ConvexHull and AlphaShape on the MPASE-projected 2D points
        ch_mask       = convex_hull_mask(pts2d, xs, ys)
        alpha         = heuristic_alpha(pts2d)
        as_mask, _, _ = alpha_shape_mask(pts2d, xs, ys, alpha=alpha)

        for level in LEVELS:
            hdr_sp  = result["shapes"].get("hdr", {}).get(PLANE, {}).get(level, {}).get(lab)
            pf_sp   = result["shapes"].get("point_fraction", {}).get(PLANE, {}).get(level, {}).get(lab)
            zero    = np.zeros((len(ys), len(xs)), dtype=bool)
            hdr_msk = hdr_sp["mask"].astype(bool) if hdr_sp else zero
            pf_msk  = pf_sp["mask"].astype(bool)  if pf_sp  else zero

            gt = gt_masks[level]
            row = {
                "shape":       shape_name,
                "seed":        seed,
                "level":       level,
                "ConvexHull":  round(iou(ch_mask,  gt), 4),
                "AlphaShape":  round(iou(as_mask,  gt), 4),
                "HDR":         round(iou(hdr_msk,  gt), 4),
                "PF":          round(iou(pf_msk,   gt), 4),
            }
            rows.append(row)
            _log(f"    {shape_name:12} seed={seed} lv={level:3}%  "
                 f"CH={row['ConvexHull']:.3f}  AS={row['AlphaShape']:.3f}  "
                 f"HDR={row['HDR']:.3f}  PF={row['PF']:.3f}")

        # Carry vis data from seed=0 for the qualitative figure
        if seed == 0:
            vis = dict(
                xs=xs, ys=ys, pts2d=pts2d, dens_grid=dens_grid,
                gt_masks=gt_masks,
                ch_mask=ch_mask, as_mask=as_mask,
                result=result, lab=lab,
            )

    return rows, vis


# ── qualitative figure ────────────────────────────────────────────────────────

def fig_qualitative(shape_name, vis):
    """
    Grid: rows = levels, cols = 4 methods (no GT column).
    Shows sampled points + method mask contour only.
    """
    QUAL_LEVELS = [95, 96, 97, 98, 99]

    xs, ys  = vis["xs"], vis["ys"]
    pts2d   = vis["pts2d"]
    ch_mask = vis["ch_mask"]
    as_mask = vis["as_mask"]
    result  = vis["result"]
    lab     = vis["lab"]

    def _get_mpase_mask(variant, level):
        sp = result["shapes"].get(variant, {}).get(PLANE, {}).get(level, {}).get(lab)
        if sp is None:
            return np.zeros((len(ys), len(xs)), dtype=bool)
        return sp["mask"].astype(bool)

    def _draw_contour(ax, mask, color, lw=2.0, ls="-", zorder=4, smooth=False):
        if mask.sum() == 0:
            return
        field = gaussian_filter(mask.astype(float), sigma=1.5) if smooth else mask.astype(float)
        thr   = 0.3 if smooth else 0.5
        for c in find_contours(field, thr):
            px = np.interp(c[:, 1], np.arange(len(xs)), xs)
            py = np.interp(c[:, 0], np.arange(len(ys)), ys)
            ax.plot(px, py, color=color, linewidth=lw, linestyle=ls, zorder=zorder)

    cols = METHODS
    fig, axes = plt.subplots(len(QUAL_LEVELS), len(cols),
                             figsize=(3.2 * len(cols), 3.0 * len(QUAL_LEVELS)))

    ext = [xs.min(), xs.max(), ys.min(), ys.max()]

    for li, level in enumerate(QUAL_LEVELS):
        hdr_msk = _get_mpase_mask("hdr", level)
        pf_msk  = _get_mpase_mask("point_fraction", level)

        method_masks = {
            "ConvexHull": ch_mask,
            "AlphaShape": as_mask,
            "HDR":        hdr_msk,
            "PF":         pf_msk,
        }

        for ci, col in enumerate(cols):
            ax    = axes[li, ci]
            mask  = method_masks[col]
            color = PALETTE.get(col, "black")
            use_smooth = col in ("HDR", "PF")

            # sampled points — clearly visible
            ax.scatter(pts2d[:, 0], pts2d[:, 1], s=10, c="#333333", alpha=0.5, zorder=1)

            # transparent fill
            if mask.sum() > 0:
                rgba       = np.zeros((*mask.shape, 4))
                rgb        = mcolors.to_rgb(color)
                rgba[mask] = [*rgb, 0.25]
                ax.imshow(rgba, extent=ext, origin="lower", aspect="auto", zorder=2)

            # solid method contour (smoothed for HDR/PF)
            _draw_contour(ax, mask, color, lw=2.2, ls="-", zorder=4, smooth=use_smooth)

            if li == 0:
                ax.set_title(col, fontsize=8, color=color, fontweight="bold")

            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"{level}%", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Shape: {shape_name}  |  N={N_PTS} pts per replicate  |  Levels 95–99%\n"
        f"(ConvexHull & AlphaShape applied to MPASE-projected 2D points)",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"qualitative_{shape_name.lower().replace('-', '_')}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── combined qualitative figure (all shapes × all methods) ───────────────────

def fig_combined_qualitative(all_vis, level=95):
    """
    One figure: rows = shapes, cols = [Points] + methods.
    First column shows only the sampled points (no mask).
    Remaining 4 columns show each method's mask at `level`%.
    HDR and PF contours are Gaussian-smoothed.
    """
    cols        = ["Points"] + METHODS
    shape_names = list(SHAPES.keys())
    n_rows, n_cols = len(shape_names), len(cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.0 * n_rows))

    for ri, shape_name in enumerate(shape_names):
        vis     = all_vis[shape_name]
        xs, ys  = vis["xs"], vis["ys"]
        pts2d   = vis["pts2d"]
        ch_mask = vis["ch_mask"]
        as_mask = vis["as_mask"]
        result  = vis["result"]
        lab     = vis["lab"]

        def _mpase_mask(variant):
            sp = result["shapes"].get(variant, {}).get(PLANE, {}).get(level, {}).get(lab)
            if sp is None:
                return np.zeros((len(ys), len(xs)), dtype=bool)
            return sp["mask"].astype(bool)

        method_masks = {
            "ConvexHull": ch_mask,
            "AlphaShape": as_mask,
            "HDR":        _mpase_mask("hdr"),
            "PF":         _mpase_mask("point_fraction"),
        }

        ext = [xs.min(), xs.max(), ys.min(), ys.max()]

        for ci, col in enumerate(cols):
            ax    = axes[ri, ci]
            color = PALETTE.get(col, "black")

            # always show points
            ax.scatter(pts2d[:, 0], pts2d[:, 1],
                       s=10, c="#333333", alpha=0.55, zorder=3)

            if col != "Points":
                mask       = method_masks[col]
                use_smooth = col in ("HDR", "PF")

                if mask.sum() > 0:
                    rgba       = np.zeros((*mask.shape, 4))
                    rgb        = mcolors.to_rgb(color)
                    rgba[mask] = [*rgb, 0.25]
                    ax.imshow(rgba, extent=ext, origin="lower", aspect="auto", zorder=2)

                    field = gaussian_filter(mask.astype(float), sigma=1.5) if use_smooth else mask.astype(float)
                    thr   = 0.3 if use_smooth else 0.5
                    for c in find_contours(field, thr):
                        px = np.interp(c[:, 1], np.arange(len(xs)), xs)
                        py = np.interp(c[:, 0], np.arange(len(ys)), ys)
                        ax.plot(px, py, color=color, linewidth=1.8, zorder=4)

            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("white")

            if ci == 0:
                ax.set_ylabel(shape_name, fontsize=9, fontweight="bold")
            if ri == 0:
                title_color = "#333333" if col == "Points" else color
                ax.set_title(col, fontsize=8, color=title_color, fontweight="bold")

    fig.suptitle(
        f"Synthetic evaluation — {level}% density level\n"
        f"Rows: shape  |  First col: raw points  |  Other cols: method mask",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"combined_qualitative_{level}pct.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── summary figures ───────────────────────────────────────────────────────────

def fig_iou_vs_level(df):
    shapes = list(SHAPES.keys())
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.5 * len(shapes), 4), sharey=True)
    for ax, shape in zip(axes, shapes):
        sub = df[df["shape"] == shape]
        for method in METHODS:
            means = sub.groupby("level")[method].mean()
            ax.plot(means.index, means.values, marker="o",
                    color=PALETTE[method], linewidth=2, label=method)
            for lv, v in means.items():
                ax.annotate(f"{v:.2f}", (lv, v),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=6.5, color=PALETTE[method])
        ax.set_title(shape, fontsize=10, fontweight="bold")
        ax.set_xlabel("Density level (%)")
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("IoU vs ground truth")
            ax.legend(fontsize=8)
    fig.suptitle(f"IoU vs ground truth density level — {N_PTS} pts/replicate, "
                 f"MPASE pipeline ({N_BOOT} boot)", fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "iou_vs_level.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_mean_iou_heatmap(df):
    sub   = df[df["level"] < 100]
    pivot = sub.groupby("shape")[METHODS].mean().reindex(list(SHAPES.keys()))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.2, vmax=0.9, ax=ax, linewidths=0.8,
                cbar_kws={"label": "Mean IoU vs ground truth (levels 95–99%)"})
    ax.set_title("Mean IoU vs ground truth (60–95% levels)\n"
                 "Evaluated via full MPASE pipeline on 3D point clouds",
                 fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "mean_iou_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_advantage_heatmap(df):
    pivot_hdr = df.groupby(["shape", "level"])["HDR"].mean().unstack()
    pivot_ch  = df.groupby(["shape", "level"])["ConvexHull"].mean().unstack()
    advantage = (pivot_hdr - pivot_ch).reindex(list(SHAPES.keys()))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(advantage, annot=True, fmt="+.3f", cmap="RdYlGn",
                center=0, vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.8,
                cbar_kws={"label": "HDR IoU − ConvexHull IoU"})
    ax.set_title("HDR advantage over ConvexHull per shape and level\n"
                 "Green = MPASE better captures the true shape",
                 fontsize=10)
    ax.set_xlabel("Density level (%)"); ax.set_ylabel("")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "hdr_advantage_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


def fig_boxplot_summary(df):
    sub    = df[df["level"] < 100]
    melted = sub.melt(id_vars=["shape", "seed", "level"],
                      value_vars=METHODS, var_name="method", value_name="IoU")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="shape", y="IoU", hue="method",
                palette=PALETTE, ax=ax)
    ax.set_title(f"IoU vs ground truth — levels 95–99%, MPASE pipeline", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("IoU vs ground truth")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Method", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "boxplot_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    _log(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    t_total  = time.time()
    all_rows = []
    all_vis  = {}

    for shape_name, density_fn in SHAPES.items():
        _log(f"\n{'='*55}")
        _log(f"  {shape_name}")
        _log(f"{'='*55}")
        t0 = time.time()

        rows, vis = evaluate_shape(shape_name, density_fn)
        all_rows.extend(rows)
        all_vis[shape_name] = vis
        _log(f"  Done — {time.time()-t0:.1f}s")

        _log(f"  Generating qualitative figure...")
        fig_qualitative(shape_name, vis)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RES_DIR, "synthetic_results.csv"), index=False)
    print("Saved synthetic_results.csv")

    summary = df.groupby(["shape", "level"])[METHODS].mean()
    summary.to_csv(os.path.join(RES_DIR, "summary_by_shape_level.csv"))
    _log("Saved summary_by_shape_level.csv")

    _log("\nGenerating combined qualitative figures (95–100%)...")
    for lv in [95, 96, 97, 98, 99, 100]:
        fig_combined_qualitative(all_vis, level=lv)

    _log("\nGenerating summary figures...")
    fig_iou_vs_level(df)
    fig_mean_iou_heatmap(df)
    fig_advantage_heatmap(df)
    fig_boxplot_summary(df)

    sub = df[df["level"] < 100]
    mean_by_shape = sub.groupby("shape")[METHODS].mean().reindex(list(SHAPES.keys()))
    overall = sub[METHODS].mean()

    _log(f"\n  === Mean IoU (levels 95–99%) ===")
    _log(mean_by_shape.to_string())
    _log(f"\n  === Overall mean across all shapes ===")
    _log(overall.to_string())
    _log(f"\nDone — {time.time()-t_total:.1f}s total")


if __name__ == "__main__":
    main()
