import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, TypedDict, Literal,List
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
    sample_icp: Optional[int] = 50000

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
    #reproducibility control
    rng_seed: int = 0 

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
    labels: List[str]                        # e.g. ["A", "B", "C"]
    aligned_points: List[np.ndarray]         # one (N,3) array per label, same order as labels
    ids_by_label: Dict[str, np.ndarray]      # label -> original row IDs
    shapes: Dict[Variant, Dict[Plane, Dict[int, Dict[str, ShapeProduct]]]]
    metrics: pd.DataFrame   # columns: plane, variant, level, A, B, IoU, meanNN, Hausdorff
    meta: dict
    background: Dict[Plane, np.ndarray]      # union-of-points mask per plane (bool [ny,nx])
    background_by_label: Dict[str, Dict[Plane, np.ndarray]]
    densities: Optional[Dict[str, Dict[Plane, np.ndarray]]]
    projections: Dict[Plane, Dict[str, np.ndarray]]