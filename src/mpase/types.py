import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, TypedDict, Literal
from dataclasses import dataclass, field

# Config with default values
# pipeline will use these values unless other values are provided in the main function
#################### Configs ####################
@dataclass
class CfgCommon:
    grid_base: int = 160
    pad_frac: float = 0.05
    # Alignment (PCA + robust ICP)
    # fraction of worst matches to ignore in each ICP iteration
    # at most 0.5 (0.0 = use all points)
    trim_q: float = 0.10
    # number of ICP iterations
    # error usually stabilizes after ~20
    icp_iters: int = 30 # use ~50 for more difficult cases
    # max points to sample from each set for ICP (None = use all)
    sample_icp: int = 50000

@dataclass
class CfgHDR:
    # Bootstrap for density averaging
    n_boot: int = 256
    sample_frac: float = 1.0 # draw n with replacement if 1.0
    # 2D density smoothing (pixels of the 2D hist grid)
    sigma_px: float = 1.2
    # Tail trimming for HDR (zeros tiny densities before threshold search)
    density_floor_frac: float = 0.002
    # Coverage (HDR) levels
    mass_levels: Tuple[float, ...] = (1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.80, 0.60, 0.50)
    # reproducibility control for bootstrap
    rng_seed: int = 0

@dataclass
class CfgMorph:
    closing: int = 1
    opening: int = 1
    keep_largest: bool = False
    fill_holes: bool = True
    
@dataclass
class CfgPF:
    frac_levels: Tuple[float, ...] = (1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.80, 0.60, 0.50)
    bandwidth: Optional[float] = None
    disk_px: int = 2
    # expose morphology controls to the user
    morph: CfgMorph = field(default_factory=CfgMorph)

# ############################ Public Types ############################
Plane = Literal["XY", "YZ", "XZ"]
Variant = Literal["hdr", "point_fraction"]

class ShapeProduct(TypedDict):
    plane: Plane
    level: int              # 50,60,80,95,100
    variant: Variant
    mask: np.ndarray        # bool [ny, nx]
    contour: Optional[np.ndarray]  # [N,2] (row, col)

class RunResult(TypedDict):
    A: np.ndarray           # aligned & scaled 3D points (N,3)
    B: np.ndarray
    shapes: Dict[Variant, Dict[Plane, Dict[int, Tuple[ShapeProduct, ShapeProduct]]]]
    metrics: pd.DataFrame   # columns: plane, level, variant, IoU, meanNN, Hausdorff
    meta: dict
    background: Dict[Plane, np.ndarray]  # union-of-points mask per plane (bool [ny,nx])
    densities: Optional[Dict[str, Dict[Plane, np.ndarray]]]  # None if HDR not run; else {'A':{plane:D}, 'B':{...}}
    projections: Dict[Plane, Dict[str, np.ndarray]]  # {"XY":{"xs","ys","A2","B2"}, ...}