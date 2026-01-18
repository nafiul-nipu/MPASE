import numpy as np
from typing import Optional

from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

# Internal dependencies
from .types import Plane, ShapeProduct
from .create_grid_planes import AXPAIR

#################### HDR ############################
# Density thresholding & contours
def apply_density_floor(D: np.ndarray, frac: float) -> np.ndarray:
    # zero out tiny densities below a fraction of the max density
    if frac <= 0: return D
    # fraction of max density
    eps = float(frac) * float(D.max() if np.isfinite(D.max()) else 0.0)
    # if the fraction is too small, return the original array
    if eps <= 0: return D
    # otherwise zero out values below eps
    Df = D.copy()
    Df[Df < eps] = 0.0
    return Df

def mass_threshold(D: np.ndarray, mass: float) -> float:
    """
    Return density threshold (tau) so that sum(D[D>=tau]) ≈ mass * sum(D).
    Special-case mass=1.0 -> choose the smallest strictly positive tau to include all positive density.
    """
    # make density map 1-D so its easy to sort and sum
    flat = D.ravel()
    # total density in the map
    # if zero/negative/NaN, return inf
    s = flat.sum()
    if s <= 0: return float('inf')
    # special case for p100 : take the smallest positive density value
    # so that all positive density is included
    # if no positive values, return inf
    if mass >= 0.999999:
        pos = flat[flat > 0]
        return float(pos.min()) if pos.size else float('inf')
    # sort densities in descending order (highest to lowest)
    v = np.sort(flat)[::-1]
    # cumulative sum of sorted densities
    c = np.cumsum(v)
    # target total = mass * (total density)
    # find the smallest index where cumulative sum >= target total
    # return the density value at that index as the threshold
    return float(v[np.searchsorted(c, mass * c[-1], side='left')])

# find contour at given mass level
def contour_at_mass(D: np.ndarray, mass: float) -> Optional[np.ndarray]:
    # get the cutoff tau, if tau is inf or NaN, return None
    tau = mass_threshold(D, mass)
    if not np.isfinite(tau): return None
    # Use skimage.measure.find_contours to trace level-tau isolines on the 2D array.
    # returns a list of polylines (each is [N,2] as [row,col])
    cs = find_contours(D, level=tau)  # D is [rows, cols]
    # if no contours found, return None
    if not cs: return None
    # Sort by number of points (proxy for length) and keep the longest contour (main silhouette).
    cs.sort(key=lambda c: c.shape[0], reverse=True)
    return cs[0]

# Bootstrap 2D densities
# resample points many times with replacement
# smooth them onto the shared grid to see where points consistently fall
# from this probability map extract regions that are always there (high-density contours)
def boot_density_2d(P: np.ndarray, edges3d, n_boot=256, sample_frac=1.0, sigma_px=1.2, rng_seed: int = 0):
    """
    Average 2D densities over bootstrap resamples.
    Returns a dict: Densities 'x|y|z' -> [ny,nx] float
    """
    # which 2D plane corresponds to which axes
    xed, yed, zed = edges3d
    plane_edges = {'z': (xed, yed), 'x': (yed, zed), 'y': (xed, zed)}
    # accumulators for density sums
    accD = {a: None for a in ('x','y','z')}
    # number of points to sample in each bootstrap resample
    n = len(P); k = max(1, int(round(sample_frac * n)))
    # random number generator
    rs = np.random.default_rng(rng_seed)
    
    for _ in range(n_boot):
        # pick k points with replacement
        idx = rs.integers(0, n, size=k) # with replacement
        sub = P[idx]
        for a in ('x','y','z'):
            # take two coords indices
            i, j = AXPAIR[a]
            ex, ey = plane_edges[a]
            # 2D histogram
            H, _, _ = np.histogram2d(sub[:, i], sub[:, j], bins=[ex, ey]) # shape [nx, ny]
            # density (smoothed) gaussian filter
            D = gaussian_filter(H.astype(np.float32), sigma=sigma_px)
            # transpose to standard orientation [rows, cols] = [y, x]
            D = D.T # -> [ny, nx]
            if accD[a] is None: accD[a] = D
            else: accD[a] += D

    # average density maps across bootstraps
    outD = {a: (accD[a] / float(n_boot)) if accD[a] is not None else None for a in accD}
    return outD

def make_hdr_shape(D: np.ndarray, plane: Plane, mass: float, density_floor_frac: float) -> ShapeProduct:
    # densities with floor (for nicer HDR contours)
    Df = apply_density_floor(D, density_floor_frac)
    # Find the density cutoff (tau) for A and B that encloses m × 100% of the mass.
    tau = mass_threshold(Df, mass)
    # Turn the density maps into binary masks 
    mask = (Df >= tau)
    # Extract the contour lines (the actual silhouette boundaries).
    contour = contour_at_mass(Df, mass)
    return dict(plane=plane, level=int(round(mass*100)), variant="hdr", mask=mask, contour=contour)
