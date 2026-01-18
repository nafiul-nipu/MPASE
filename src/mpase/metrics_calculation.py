import numpy as np
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from skimage.measure import find_contours
##################### Common metrics #####################
# Metrics 
# Intersection-over-Union (IoU) for boolean masks
def iou_bool(A: np.ndarray, B: np.ndarray) -> float:
    # if either is None or empty, return NaN
    inter = np.logical_and(A,B).sum()
    union = np.logical_or(A,B).sum()
    return float(inter) / float(union + 1e-9)

# Contour distances (mean nearest neighbor + Hausdorff)
# using cKDTree for fast nearest neighbor search
def contour_distances(CA: Optional[np.ndarray], CB: Optional[np.ndarray]) -> Tuple[float,float]:
    if CA is None or CB is None: return float('nan'), float('nan')
    TA, TB = cKDTree(CA), cKDTree(CB)
    da,_ = TA.query(CB, k=1); db,_ = TB.query(CA, k=1)
    return float((da.mean()+db.mean())/2.0), float(max(da.max(), db.max()))

# find contour from boolean mask
def contour_from_bool(M: np.ndarray) -> Optional[np.ndarray]:
    if M is None or M.sum() == 0: return None
    # find contours at level 0.5 (between False=0 and True=1)
    cs = find_contours(M.astype(float), level=0.5)
    if not cs: return None
    # return the longest contour
    cs.sort(key=lambda c: c.shape[0], reverse=True)
    return cs[0]

def _poly_area_rc(poly: np.ndarray) -> float:
    """
    Polygon area from contour in [row, col] coords.
    We treat (x,y) = (col,row) and use shoelace formula.
    """
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 1]
    y = poly[:, 0]
    return 0.5 * float(
        np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )

def all_contours_from_bool(
    M: np.ndarray,
    min_len: int = 10,
    min_area_frac: float = 0.0,
) -> list[np.ndarray]:
    """
    Return all reasonably sized contour loops from a boolean mask.

    - min_len: drop contours with too few vertices (short noisy wiggles)
    - min_area_frac:
        * <= 0  → no area filtering (keep all lengths >= min_len)
        * > 0   → keep only contours whose area >= min_area_frac * max_area
    """
    if M is None or M.sum() == 0:
        return []

    cs = find_contours(M.astype(float), level=0.5)
    if not cs:
        return []

    # 1) length filter
    cs = [c for c in cs if c.shape[0] >= min_len]
    if not cs:
        return []

    # 2) optional area filter
    if min_area_frac <= 0:
        return cs

    areas = np.array([_poly_area_rc(c) for c in cs], dtype=float)
    max_area = float(areas.max()) if areas.size else 0.0
    if max_area <= 0:
        return cs

    keep = [
        c for c, a in zip(cs, areas)
        if a >= min_area_frac * max_area
    ]
    return keep
