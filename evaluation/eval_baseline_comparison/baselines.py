"""
Baseline shape methods: Convex Hull and Alpha Shape.
Both are rasterized onto the shared MPASE projection grid so metrics are directly comparable.

Alpha shape uses a KDTree-based heuristic for alpha (1 / (2 * median_nn_distance)),
which is instant and produces meaningful concave hulls without the slow optimizealpha
binary search that tends to degenerate back to convex hull anyway.
"""

import numpy as np
from typing import Optional, Tuple
from matplotlib.path import Path
from scipy.spatial import ConvexHull, KDTree
from skimage.measure import find_contours


# ── rasterization ─────────────────────────────────────────────────────────────

def _rasterize_paths(paths: list, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon for a list of matplotlib Paths onto grid."""
    ny, nx = len(ys), len(xs)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.zeros(ny * nx, dtype=bool)
    for p in paths:
        try:
            mask |= p.contains_points(grid_pts)
        except Exception:
            pass
    return mask.reshape(ny, nx)


# ── convex hull ───────────────────────────────────────────────────────────────

def convex_hull_mask(pts2d: np.ndarray,
                     xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
    """Binary mask: convex hull of pts2d rasterized onto (ny × nx) grid."""
    ny, nx = len(ys), len(xs)
    if len(pts2d) < 3:
        return np.zeros((ny, nx), dtype=bool)
    try:
        hull  = ConvexHull(pts2d)
        verts = pts2d[hull.vertices]
        verts = np.vstack([verts, verts[0]])   # close polygon
        return _rasterize_paths([Path(verts)], xs, ys)
    except Exception:
        return np.zeros((ny, nx), dtype=bool)


# ── alpha shape ───────────────────────────────────────────────────────────────

def heuristic_alpha(pts2d: np.ndarray) -> float:
    """
    Estimate alpha from the median nearest-neighbour distance in the point set.
    alpha = 1 / (2 * median_nn_dist)
    This is O(N log N) and avoids the slow optimizealpha binary search.
    """
    if len(pts2d) < 4:
        return 0.0
    tree  = KDTree(pts2d)
    dists, _ = tree.query(pts2d, k=2)   # k=2: point itself + nearest neighbour
    med   = np.median(dists[:, 1])
    if med <= 0:
        return 0.0
    return 1.0 / (2.0 * med)


def alpha_shape_mask(pts2d: np.ndarray,
                     xs: np.ndarray,
                     ys: np.ndarray,
                     alpha: float = None) -> Tuple[np.ndarray, float, bool]:
    """
    Binary mask: alpha shape rasterized onto grid.
    Returns (mask, alpha_used, fell_back_to_convex_hull).

    If alpha is None, heuristic_alpha() is used (fast KDTree estimate).
    Falls back to convex hull if the shape is empty or invalid.
    """
    import alphashape
    from shapely.geometry import MultiPolygon, Polygon

    ny, nx = len(ys), len(xs)
    zero   = np.zeros((ny, nx), dtype=bool)

    if len(pts2d) < 4:
        return zero, 0.0, True

    if alpha is None:
        alpha = heuristic_alpha(pts2d)

    try:
        shape = alphashape.alphashape(pts2d, float(alpha))
    except Exception:
        return convex_hull_mask(pts2d, xs, ys), alpha, True

    if shape is None or shape.is_empty:
        return convex_hull_mask(pts2d, xs, ys), alpha, True

    if isinstance(shape, Polygon):
        polys = [shape]
    elif isinstance(shape, MultiPolygon):
        polys = list(shape.geoms)
    else:
        return zero, alpha, True

    paths = []
    for poly in polys:
        if poly.is_empty or len(poly.exterior.coords) < 3:
            continue
        paths.append(Path(np.array(poly.exterior.coords)))

    if not paths:
        return convex_hull_mask(pts2d, xs, ys), alpha, True

    return _rasterize_paths(paths, xs, ys), alpha, False


# ── contour utilities ─────────────────────────────────────────────────────────

def mask_to_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Longest contour from binary mask in (row, col) pixel coordinates."""
    if mask is None or mask.sum() == 0:
        return None
    cs = find_contours(mask.astype(float), level=0.5)
    if not cs:
        return None
    cs.sort(key=lambda c: c.shape[0], reverse=True)
    return cs[0]


def contour_to_physical(contour_rc: np.ndarray,
                         xs: np.ndarray,
                         ys: np.ndarray) -> np.ndarray:
    """Convert (row, col) contour to physical (x, y) grid coordinates."""
    phys_x = np.interp(contour_rc[:, 1], np.arange(len(xs)), xs)
    phys_y = np.interp(contour_rc[:, 0], np.arange(len(ys)), ys)
    return np.column_stack([phys_x, phys_y])
