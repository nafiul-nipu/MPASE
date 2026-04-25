"""
Synthetic Shape Evaluation — Direct MPASE API
=============================================
Calls MPASE internal functions directly (no mpase.run()).
All seeds share a canonical grid built from 10 k large samples.

Five shapes: Ellipse, Two-lobe, Crescent, Asymmetric, Ring
Four methods: ConvexHull, AlphaShape (circumradius-based), HDR, PF

Evaluations:
  - Stability:        pairwise IoU across 10 random seeds at 100 %
  - Discriminability: IoU between original and 3 perturbation types

Outputs (./synthetic_eval/):
    stability_metrics.csv
    discriminability_metrics.csv
    heatmap_stability_iou.png
    heatmap_discriminability.png
    bar_discrimination_gap.png
    qualitative_60pct.png
    qualitative_100pct.png

Usage:
    cd /path/to/MPASE
    python -u evaluation/eval_synthetic_shapes/run.py 2>&1 | tee /tmp/synth_shapes.log
"""

import os, sys, itertools, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import gaussian_filter, binary_closing as _bin_closing
from skimage.draw import polygon as sk_polygon
from skimage.measure import find_contours
import seaborn as sns

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mpase.create_grid_planes  import make_grid_from_bounds, grid_centers_from_edges
from mpase.hdr_bootstrap       import boot_density_2d, make_hdr_shape
from mpase.point_fraction      import make_pf_shape, kde_scores, auto_bandwidth
from mpase.metrics_calculation import iou_bool, contour_from_bool
from mpase.types               import CfgMorph

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(_DIR, "synthetic_eval")
os.makedirs(OUT, exist_ok=True)

def _log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── constants ─────────────────────────────────────────────────────────────────
LEVELS_FULL = [100, 60]     # levels used in discriminability
N_PTS    = 500
N_LARGE  = 10_000           # for canonical grid
N_BOOT   = 256
N_SEEDS  = 10               # stability
Z_NOISE  = 0.001
BASE     = 160

# Same HDR/PF parameters as eval_synthetic/run.py
SIGMA_PX   = 3.0
FLOOR_FRAC = 0.01
DISK_PX    = 3
PF_MORPH   = CfgMorph(closing=3, opening=1, keep_largest=False, fill_holes=False)

METHODS = ["ConvexHull", "AlphaShape", "HDR", "PF"]
PALETTE = {
    "Points":      "#777777",
    "ConvexHull":  "#e15759",
    "AlphaShape":  "#f28e2b",
    "HDR":         "#4e79a7",
    "PF":          "#59a14f",
}

PERTURB_TYPES = ["Shift", "Disperse", "RemoveRegion"]

# ── shape definitions ─────────────────────────────────────────────────────────

class Shape:
    def __init__(self, name, sampler_fn):
        self.name     = name
        self._sampler = sampler_fn   # (n, rng) -> (n, 2)

    def sample(self, n, rng, noise_std=0.03):
        pts = self._sampler(n, rng)
        return pts + rng.normal(0, noise_std, pts.shape)


def _ell_sampler(n, rng):
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-1.5, 1.5, n * 3)
        y = rng.uniform(-0.8, 0.8, n * 3)
        ok = (x / 1.0) ** 2 + (y / 0.5) ** 2 <= 1.0
        pts.append(np.c_[x[ok], y[ok]])
    return np.vstack(pts)[:n]


def _two_sampler(n, rng):
    h = n // 2
    a = rng.multivariate_normal([-0.6, 0], [[0.08, 0], [0, 0.05]], h)
    b = rng.multivariate_normal([ 0.6, 0], [[0.08, 0], [0, 0.05]], n - h)
    return np.vstack([a, b])


def _cres_sampler(n, rng):
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.uniform(-1.2, 1.2, n * 4)
        y = rng.uniform(-1.2, 1.2, n * 4)
        ok = (x ** 2 + y ** 2 <= 1.0) & ((x - 0.3) ** 2 + y ** 2 > 0.49)
        pts.append(np.c_[x[ok], y[ok]])
    return np.vstack(pts)[:n]


def _asym_sampler(n, rng):
    n1 = int(0.8 * n)
    a  = rng.multivariate_normal([0,   0  ], [[0.20, 0], [0, 0.15]], n1)
    b  = rng.multivariate_normal([1.0, 0.5], [[0.03, 0], [0, 0.03]], n - n1)
    return np.vstack([a, b])


def _ring_sampler(n, rng):
    theta = rng.uniform(0, 2 * np.pi, n)
    r     = np.sqrt(rng.uniform(0.6 ** 2, 1.0 ** 2, n))
    return np.c_[r * np.cos(theta), r * np.sin(theta)]


SHAPES = [
    Shape("Ellipse",    _ell_sampler),
    Shape("Two-lobe",   _two_sampler),
    Shape("Crescent",   _cres_sampler),
    Shape("Asymmetric", _asym_sampler),
    Shape("Ring",       _ring_sampler),
]

# ── canonical grid ────────────────────────────────────────────────────────────

_CACHE = {}

def _get_cache(shape):
    """Canonical grid built from a large sample — shared across all seeds."""
    if shape.name not in _CACHE:
        rng  = np.random.default_rng(999)
        p2d  = shape.sample(N_LARGE, rng, noise_std=0.01)
        p3d  = np.c_[p2d, rng.normal(0, Z_NOISE, N_LARGE)]
        edges, _ = make_grid_from_bounds(p3d, base=BASE, pad_frac=0.15)
        xs, ys   = grid_centers_from_edges(edges[0], edges[1])
        _CACHE[shape.name] = dict(edges=edges, xs=xs, ys=ys)
    return _CACHE[shape.name]

# ── perturbation generators ───────────────────────────────────────────────────

def perturb(pts2d, kind, rng):
    n   = len(pts2d)
    k50 = int(0.5 * n)

    if kind == "Shift":
        # move half the points by 0.6 units — larger than sampling noise for N=500
        out = pts2d.copy()
        idx = rng.choice(n, size=k50, replace=False)
        out[idx, 0] += 0.6
        return out

    if kind == "Disperse":
        # scatter half the points outward — clearly beyond sampling variability
        out = pts2d.copy()
        idx = rng.choice(n, size=k50, replace=False)
        out[idx] += rng.normal(0, 0.4, (k50, 2))
        return out

    if kind == "RemoveRegion":
        keep = ~((pts2d[:, 0] > 0) & (pts2d[:, 1] > 0))
        remaining = pts2d[keep]
        n_need = n - len(remaining)
        if n_need > 0 and len(remaining) >= 2:
            extra = remaining[rng.choice(len(remaining), size=n_need, replace=True)]
            return np.vstack([remaining, extra])
        return remaining[:n] if len(remaining) >= n else remaining

    raise ValueError(kind)

# ── baseline helpers ──────────────────────────────────────────────────────────

def _density_subset(pts2d, frac):
    """Top-frac fraction of points by KDE density (mirrors PF logic)."""
    if frac >= 1.0:
        return pts2d
    bw     = auto_bandwidth(pts2d)
    scores = kde_scores(pts2d, bw)
    k      = max(3, int(np.ceil(frac * len(pts2d))))
    return pts2d[np.argsort(scores)[::-1][:k]]


def _rasterise_polygon(verts_xy, xs, ys):
    ny, nx = len(ys), len(xs)
    XX, YY = np.meshgrid(xs, ys)
    path   = MplPath(verts_xy)
    return path.contains_points(np.c_[XX.ravel(), YY.ravel()]).reshape(ny, nx)


def baseline_convex_hull(pts2d, xs, ys, frac):
    sub = _density_subset(pts2d, frac)
    if len(sub) < 3:
        return np.zeros((len(ys), len(xs)), bool), None
    try:
        hull = ConvexHull(sub)
        mask = _rasterise_polygon(sub[hull.vertices], xs, ys)
        return mask, contour_from_bool(mask)
    except Exception:
        return np.zeros((len(ys), len(xs)), bool), None


def _circumradius(pa, pb, pc):
    """Circumradius for each triangle (pa, pb, pc each [N, 2])."""
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pa - pc, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)
    # twice the signed area (absolute value)
    area2 = np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                 - (pc[:, 0] - pa[:, 0]) * (pb[:, 1] - pa[:, 1]))
    return (a * b * c) / (area2 + 1e-12)


def baseline_alpha_shape(pts2d, xs, ys, frac):
    """Alpha shape via Delaunay + circumradius filter (75th-percentile threshold)."""
    sub    = _density_subset(pts2d, frac)
    ny, nx = len(ys), len(xs)
    mask   = np.zeros((ny, nx), bool)
    if len(sub) < 4:
        return mask, None
    try:
        tri = Delaunay(sub)
        ia, ib, ic = tri.simplices.T
        R    = _circumradius(sub[ia], sub[ib], sub[ic])
        # 75th-percentile circumradius as alpha — adapts to local point density
        alpha = np.percentile(R[np.isfinite(R)], 75)
        keep  = R <= alpha
        for simp in tri.simplices[keep]:
            v    = sub[simp]
            cols = np.clip(np.searchsorted(xs, v[:, 0], side='right') - 1, 0, nx - 1)
            rows = np.clip(np.searchsorted(ys, v[:, 1], side='right') - 1, 0, ny - 1)
            rr, cc = sk_polygon(rows, cols, (ny, nx))
            mask[rr, cc] = True
        # fill inter-triangle gaps — 20 iterations closes gaps up to ~20 px
        # while preserving large structural holes (ring centre, crescent inner circle)
        mask = _bin_closing(mask, iterations=20)
        return mask, contour_from_bool(mask)
    except Exception:
        return mask, None

# ── core method runner ────────────────────────────────────────────────────────

def run_all_methods(pts2d, pts3d, edges, xs, ys, levels):
    """dict[method][level] = (mask, contour) for all methods."""
    out = {m: {} for m in METHODS}

    # HDR: bootstrap density once, threshold at each level
    densities = boot_density_2d(pts3d, edges, n_boot=N_BOOT,
                                sigma_px=SIGMA_PX, rng_seed=0)
    D_xy = densities['z']
    for lv in levels:
        sp = make_hdr_shape(D_xy, 'XY', lv / 100.0, FLOOR_FRAC)
        out["HDR"][lv] = (sp['mask'], sp['contour'])

    # PF
    for lv in levels:
        sp = make_pf_shape(pts2d, xs, ys, 'XY', lv / 100.0,
                           bandwidth=None, disk_px=DISK_PX, morph=PF_MORPH)
        out["PF"][lv] = (sp['mask'], sp['contour'])

    # ConvexHull + AlphaShape
    for lv in levels:
        frac = lv / 100.0
        out["ConvexHull"][lv] = baseline_convex_hull(pts2d, xs, ys, frac)
        out["AlphaShape"][lv] = baseline_alpha_shape(pts2d, xs, ys, frac)

    return out

# ── stability evaluation ──────────────────────────────────────────────────────

def run_stability():
    rows = []
    for shape in SHAPES:
        _log(f"Stability {shape.name}")
        c = _get_cache(shape)

        seed_masks = {m: [] for m in METHODS}
        for seed in range(N_SEEDS):
            rng   = np.random.default_rng(seed)
            pts2d = shape.sample(N_PTS, rng)
            pts3d = np.c_[pts2d, rng.normal(0, Z_NOISE, N_PTS)]
            res   = run_all_methods(pts2d, pts3d, c['edges'], c['xs'], c['ys'], [100])
            for m in METHODS:
                seed_masks[m].append(res[m][100][0])

        for method in METHODS:
            masks = seed_masks[method]
            ious  = [iou_bool(masks[i], masks[j])
                     for i, j in itertools.combinations(range(N_SEEDS), 2)]
            rows.append(dict(
                shape=shape.name, method=method, level=100,
                mean_pairwise_iou=round(float(np.mean(ious)), 4),
                std_pairwise_iou=round(float(np.std(ious)), 4),
            ))
    return pd.DataFrame(rows)

# ── discriminability evaluation ───────────────────────────────────────────────

def run_discriminability():
    rows = []
    for shape in SHAPES:
        _log(f"Discriminability {shape.name}")
        c = _get_cache(shape)

        # original (same seed as qualitative)
        rng_orig  = np.random.default_rng(42)
        orig2d    = shape.sample(N_PTS, rng_orig)
        orig3d    = np.c_[orig2d, rng_orig.normal(0, Z_NOISE, N_PTS)]
        orig_res  = run_all_methods(orig2d, orig3d, c['edges'], c['xs'], c['ys'], LEVELS_FULL)

        for pert_kind in PERTURB_TYPES:
            rng_p  = np.random.default_rng({"Shift": 77, "Disperse": 78, "RemoveRegion": 79}[pert_kind])
            pert2d = perturb(orig2d, pert_kind, rng_p)
            pert3d = np.c_[pert2d, rng_p.normal(0, Z_NOISE, len(pert2d))]
            pert_res = run_all_methods(pert2d, pert3d, c['edges'], c['xs'], c['ys'], LEVELS_FULL)

            for method in METHODS:
                for lv in LEVELS_FULL:
                    orig_mask = orig_res[method][lv][0]
                    pert_mask = pert_res[method][lv][0]
                    iou_val   = iou_bool(orig_mask, pert_mask)
                    rows.append(dict(
                        shape=shape.name,
                        perturbation=pert_kind,
                        method=method,
                        level=lv,
                        iou_original_vs_perturbed=round(float(iou_val), 4),
                    ))
    return pd.DataFrame(rows)

# ── figures ───────────────────────────────────────────────────────────────────

def _draw_contours(ax, mask, xs, ys, color, smooth=False, lw=2.2, min_len=20):
    """
    Draw contours from a binary mask at level=0.5 (matches MPASE all_contours_from_bool).
    Loop is closed only when endpoints are within 3 px — avoids diagonal artifact
    if a contour path touches the image boundary.
    """
    if mask.sum() == 0:
        return
    if smooth:
        field, thr = gaussian_filter(mask.astype(float), sigma=1.5), 0.3
    else:
        field, thr = mask.astype(float), 0.5

    for c in find_contours(field, thr):
        if len(c) < min_len:
            continue
        px = np.interp(c[:, 1], np.arange(len(xs)), xs)
        py = np.interp(c[:, 0], np.arange(len(ys)), ys)
        if np.linalg.norm(c[-1] - c[0]) < 3:   # close only genuine interior loops
            px = np.append(px, px[0])
            py = np.append(py, py[0])
        ax.plot(px, py, color=color, lw=lw, zorder=4,
                solid_capstyle='round', solid_joinstyle='round')


def make_qualitative(level=60):
    _log(f"Qualitative figure at {level}%")
    cols = ["Points"] + METHODS

    # display labels (α-Shape avoids long-title overflow)
    COL_LABELS = {"Points": "Points", "ConvexHull": "ConvexHull",
                  "AlphaShape": "α-Shape", "HDR": "HDR", "PF": "PF"}

    fig, axes = plt.subplots(
        len(SHAPES), len(cols),
        figsize=(3.2 * len(cols), 3.2 * len(SHAPES) + 0.5),
        constrained_layout=True,
    )
    fig.suptitle(f"Synthetic evaluation — {level}% density level", fontsize=16)

    for r, shape in enumerate(SHAPES):
        c      = _get_cache(shape)
        xs, ys = c['xs'], c['ys']
        ext    = [xs.min(), xs.max(), ys.min(), ys.max()]

        rng   = np.random.default_rng(42)
        pts2d = shape.sample(N_PTS, rng)
        pts3d = np.c_[pts2d, rng.normal(0, Z_NOISE, N_PTS)]
        res   = run_all_methods(pts2d, pts3d, c['edges'], xs, ys, [level])

        for ci, col in enumerate(cols):
            ax    = axes[r, ci]
            color = PALETTE[col]
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor('white')

            if ci == 0:
                ax.set_ylabel(shape.name, fontsize=15, fontweight='bold', color='#222222')
            if r == 0:
                title_color = '#444444' if col == 'Points' else color
                ax.set_title(COL_LABELS[col], fontsize=15, color=title_color,
                             fontweight='bold', pad=8)

            # points — visible above the fill
            pt_color = '#333333' if col == 'Points' else '#555555'
            ax.scatter(pts2d[:, 0], pts2d[:, 1],
                       s=6, c=pt_color, alpha=0.55, rasterized=True, zorder=3)

            if col != 'Points':
                mask, _ = res[col][level]

                if mask.sum() > 0:
                    rgba = np.zeros((*mask.shape, 4))
                    rgb  = mcolors.to_rgb(color)
                    rgba[mask] = [*rgb, 0.25]
                    ax.imshow(rgba, extent=ext, origin='lower', aspect='auto', zorder=2)

                # match MPASE visualization: find_contours at level=0.5 on the raw binary mask
                # HDR mask is already smooth (density was gaussian-filtered at sigma=3.0 before thresholding)
                _draw_contours(ax, mask, xs, ys, color, smooth=False, lw=2.2)

    path = os.path.join(OUT, f"qualitative_{level}pct.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f"  -> {path}")


def make_stability_heatmap(stab: pd.DataFrame):
    _log("Heatmap: stability pairwise IoU")
    shape_order  = [s.name for s in SHAPES]
    method_order = METHODS

    mat     = stab.pivot(index="shape", columns="method", values="mean_pairwise_iou")
    mat     = mat.reindex(index=shape_order, columns=method_order)
    std_mat = stab.pivot(index="shape", columns="method", values="std_pairwise_iou")
    std_mat = std_mat.reindex(index=shape_order, columns=method_order)

    annot = pd.DataFrame("", index=shape_order, columns=method_order)
    for row in shape_order:
        for col in method_order:
            annot.loc[row, col] = f"{mat.loc[row, col]:.3f}\n±{std_mat.loc[row, col]:.3f}"

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.heatmap(mat, ax=ax, annot=annot, fmt="", vmin=0, vmax=1,
                cmap="YlGnBu", linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Stability: Mean Pairwise IoU at 100% (10 seeds)", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    path = os.path.join(OUT, "heatmap_stability_iou.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f"  -> {path}")


def make_discriminability_figures(disc: pd.DataFrame, stab: pd.DataFrame):
    shape_order  = [s.name for s in SHAPES]
    method_order = METHODS

    # ── heatmap: mean discriminability IoU at 100% (shapes × methods) ──────────
    _log("Heatmap: discriminability IoU")
    sub100 = disc[disc["level"] == 100]
    mat    = sub100.groupby(["shape", "method"])["iou_original_vs_perturbed"].mean().unstack("method")
    mat    = mat.reindex(index=shape_order, columns=method_order)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.heatmap(mat, ax=ax, annot=True, fmt=".3f", vmin=0, vmax=1,
                cmap="YlOrRd_r", linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Discriminability: IoU (original vs perturbed) at 100%\n"
                 "Lower = method detects the change", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    path = os.path.join(OUT, "heatmap_discriminability.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f"  -> {path}")

    # ── bar chart: discrimination gap per method ────────────────────────────────
    _log("Bar chart: discrimination gap")
    stab_mean = stab[stab["level"] == 100].groupby("method")["mean_pairwise_iou"].mean()
    disc_mean = sub100.groupby("method")["iou_original_vs_perturbed"].mean()
    gap       = (stab_mean - disc_mean).reindex(method_order)

    colors = [PALETTE[m] for m in method_order]
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    bars = ax.bar(method_order, gap.values, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    ax.axhline(0, color='#333333', lw=0.8)
    for bar, val in zip(bars, gap.values):
        ypos = val + 0.005 if val >= 0 else val - 0.005
        va   = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha='center', va=va, fontsize=9)
    ax.set_ylabel("Discrimination Gap\n(Stability IoU − Discriminability IoU)")
    ymin = min(0, float(gap.values.min())) * 1.3 - 0.02
    ymax = max(0, float(gap.values.max())) * 1.3
    ax.set_ylim(ymin, ymax)
    ax.set_title("Discrimination Gap — Higher = Better\n"
                 "(method separates real changes from sampling noise)", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    path = os.path.join(OUT, "bar_discrimination_gap.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f"  -> {path}")

# ── cleanup stale outputs ─────────────────────────────────────────────────────

def _remove_stale():
    for fname in ["fidelity_metrics.csv", "bar_mean_fidelity.png", "heatmap_fidelity_iou.png"]:
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            os.remove(p)
            _log(f"Removed stale file: {fname}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Synthetic Shape Evaluation — Direct MPASE API")
    print(f"  N_PTS={N_PTS}  N_BOOT={N_BOOT}  N_SEEDS={N_SEEDS}"
          f"  sigma_px={SIGMA_PX}  disk_px={DISK_PX}")
    print("=" * 65)

    _remove_stale()

    _log("Pre-computing canonical grids ...")
    for shape in SHAPES:
        _get_cache(shape)

    _log("Running stability evaluation ...")
    stab = run_stability()
    stab.to_csv(os.path.join(OUT, "stability_metrics.csv"), index=False)
    print("\n=== Stability: Mean Pairwise IoU at 100% ===")
    print(stab[["shape", "method", "mean_pairwise_iou", "std_pairwise_iou"]].to_string(index=False))

    _log("Running discriminability evaluation ...")
    disc = run_discriminability()
    disc.to_csv(os.path.join(OUT, "discriminability_metrics.csv"), index=False)
    print("\n=== Discriminability: Mean IoU (original vs perturbed) at 100% ===")
    piv = disc[disc["level"] == 100].groupby(["shape", "method"])["iou_original_vs_perturbed"].mean().unstack()
    print(piv.round(3).to_string())

    print("\n=== Discrimination Gap (Stability − Discriminability) at 100% ===")
    stab_mean = stab[stab["level"] == 100].groupby("method")["mean_pairwise_iou"].mean()
    disc_mean = disc[disc["level"] == 100].groupby("method")["iou_original_vs_perturbed"].mean()
    print((stab_mean - disc_mean).round(3).to_string())

    make_qualitative(level=60)
    make_qualitative(level=100)
    make_stability_heatmap(stab)
    make_discriminability_figures(disc, stab)

    _log("Done. Results saved to synthetic_eval/")
