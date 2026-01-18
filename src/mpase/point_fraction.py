import math
import numpy as np
from typing import Optional, Tuple

from scipy.spatial.distance import pdist
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, label as _label
from sklearn.neighbors import KernelDensity

# Internal dependencies
from .types import CfgMorph, Plane, ShapeProduct
from .create_grid_planes import rasterize_points
from .metrics_calculation import contour_from_bool


########################## Point-Fraction ##########################
def kde_scores(points2d: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian KDE score per point (log-density up to constant).
    We’re computing per-point density scores using Gaussian kernel density estimation (KDE). 
    The output is log-density (up to an additive constant), which is perfect for ranking points by crowdedness.
    """
    # If there are fewer than 2 points, return zeros. KDE needs at least 2 points to compute density.
    if points2d.shape[0] < 2:
        return np.zeros(points2d.shape[0])
    # Build and fit a KDE model with the specified bandwidth
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(points2d)
    # Compute the log-density at each data point
    # We only need relative order (who’s denser than whom), 
    # so log-density is ideal and numerically stable.
    return kde.score_samples(points2d)

def _scott_bandwidth(points2d: np.ndarray) -> float:
    # Scott's rule on 2D: h ~ n^(-1/(d+4)) times std; use median distance fallback if degenerate
    n = max(2, points2d.shape[0]); d = 2
    std = np.std(points2d, axis=0).mean() or 1.0
    return max(1e-6, std * (n ** (-1.0/(d+4))))
    
# Pick a data-driven KDE bandwidth when the user doesn’t specify one.
def auto_bandwidth(points2d: np.ndarray, strategy: str = "median*0.5") -> float:
    """
    strategy E {"median*0.5","scott"}
    """
    # Count points to decide subsampling and edge cases.
    N = points2d.shape[0]
    # If there are 0 or 1 points, return a default bandwidth of 1.0.
    if N <= 1:
        return 1.0
    if strategy == "scott":
        return _scott_bandwidth(points2d)
    # For large datasets, subsample 500 points without replacement.
    # Computing all pairwise distances scales as O(N^2)
    # subsampling keeps this step fast while still representative.
    if N > 500:
        idx = np.random.choice(N, size=500, replace=False)
        samp = points2d[idx]
    else:
        samp = points2d
    # Compute the median pairwise distance between sampled points.
    md = np.median(pdist(samp)) if samp.shape[0] >= 2 else 1.0
    # Heuristic bandwidth = half the median distance, clamped to a tiny positive minimum.
    return max(1e-6, md * 0.5)

def point_fraction_mask(points2d: np.ndarray,
                        xs, ys,
                        frac: float,
                        bandwidth: Optional[float],
                        disk_px: int,
                        morph: Optional[CfgMorph]=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Binary mask enclosing the top ceil(frac*N) points by local density.
    Returns (mask, kept_points2d, bandwidth_used).
    """
    # use-defined morph config or default
    morph = morph or CfgMorph()
    # Count total points
    N = points2d.shape[0]
    # Compute the number of points to keep: k = ceil(f × N), but at least 1.
    k = max(1, int(math.ceil(frac * N)))
    # Choose KDE bandwidth: use provided value or auto-estimate from the data.
    # KDE needs a scale; auto keeps things robust across datasets.
    bw = auto_bandwidth(points2d, strategy="median*0.5") if bandwidth is None else bandwidth
    # Compute a density (crowdedness) score per point via KDE.
    # We want to rank points by how densely they sit among neighbors, so outliers get low scores.
    scores = kde_scores(points2d, bandwidth=bw)
    # Get indices that sort points descending by density score.
    # Highest scores first = densest points first.
    order = np.argsort(scores)[::-1]
    # Slice the first k indices = densest k points.
    keep_idx = order[:k]
    # Extract those densest points.
    # We’ll rasterize only these and also return them for audit (len(kept) should equal k).
    kept = points2d[keep_idx]
    
    # Draw the kept points as small disks on the shared grid to create a raw boolean mask.
    # Converts scattered points into a contiguous region suitable for cleanup/contouring
    # mask = rasterize_points(kept, xs, ys, disk_px=disk_px)
    mask = rasterize_points(kept, xs, ys, disk_px=disk_px)
    if morph.closing > 0:
        # Morphological closing to seal tiny gaps and connect near pixels.
        # Prevents hairline breaks in the region.
        mask = binary_closing(mask, iterations=morph.closing)
    if morph.opening > 0:
        # Morphological opening to remove small specks/outliers.
        # Removes tiny noise pixels.
        mask = binary_opening(mask, iterations=morph.opening)
    if morph.keep_largest:
        # Keep only the largest connected region and fill holes.
        # Ensures a single solid silhouette.
        mask = biggest_component_mask(mask, fill_holes=morph.fill_holes)
    elif morph.fill_holes:
        # Fill holes in all components (rarely needed; keep_largest is typical)
        mask = binary_fill_holes(mask)
    
    return mask, kept, bw

def biggest_component_mask(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """
    Cleans a binary mask by keeping only the largest connected component and filling any holes in it.
    This produces a solid silhouette.
    It keeps only the largest connected blob (so small specks or stray fragments are removed).
    It then fills any holes inside that blob (so we get a solid region instead of a donut).
    The result is a clean silhouette mask that’s stable and interpretable.
    """
    # if the mask is empty, return it as is
    if mask.sum() == 0:
        return mask

    # label scans the mask and assigns a unique number to each connected blob of True pixels.
    # We need to know how many blobs exist, so we can keep the biggest one
    labeled_mask, num_features = _label(mask)
    if num_features == 0:
        return np.zeros_like(mask)

    # Count how many pixels belong to each label. 
    # Index 0 = background, 1+ = real blobs. 
    # If only background exists, return an empty mask.
    # This step avoids returning junk when there’s no real component.
    component_sizes = np.bincount(labeled_mask.ravel())
    if len(component_sizes) <= 1: # Only background found
        return np.zeros_like(mask)

    # Find the label of the largest blob (biggest pixel count). 
    # Adds +1 because we skipped background when indexing.
    # Ensures we only keep the main silhouette, not specks
    # Ignore component 0 (background)
    largest_component_label = np.argmax(component_sizes[1:]) + 1
    
    # Build a new mask that keeps only the largest blob. Everything else is set to False.
    # Removes noise and isolates the silhouette we care about.
    largest_component = (labeled_mask == largest_component_label)
    # Fill internal holes inside the blob. Turns donut-like shapes into solid regions.
    # Ensures a consistent, interpretable outline.
    solid_mask = binary_fill_holes(largest_component) if fill_holes else largest_component
    return solid_mask

def make_pf_shape(points2d: np.ndarray,
                  xs,
                  ys, 
                  plane: Plane, 
                  frac: float,
                  bandwidth: Optional[float], 
                  disk_px: int, 
                  morph: Optional[CfgMorph]=None) -> ShapeProduct:
    """ INPUT: 
        points2d: the 2D projected points for one condition (A or B) on this plane.
        xs, ys: the shared grid’s pixel-center coordinates (so masks line up across A/B).
        plane: 
        frac: the target fraction (e.g., 0.5, 0.8, 1.0).
        bandwidth: KDE bandwidth (or None to auto-pick).
        disk_px: disk radius in pixels used when rasterizing kept points.
        OUTPUT: 
        contour: the main contour (Nx2 array of [row, col] coordinates) or None if no contour,
        mask: the cleaned boolean mask for this fraction,
        kept: the actual subset of points retained by the fraction rule (densest ceil(frac×N)).
    """
    # build the fraction mask and get the kept points
    # mask, kept, bw = point_fraction_mask(points2d, xs, ys, frac, bandwidth, disk_px, morph=morph)
    mask, kept, bw = point_fraction_mask(points2d, xs, ys, frac, bandwidth, disk_px, morph=morph)
    # Extract the main contour from the boolean mask
    contour = contour_from_bool(mask)
    return dict(plane=plane, level=int(round(frac*100)), variant="point_fraction", mask=mask, contour=contour)
