"""
# MultiPointAlignmentShapeExtractor (MPASE)

**MultiPointAlignmentShapeExtractor (MPASE)** is a pipeline for aligning multiple 3D point clouds, extracting density- and fraction-based shapes, and generating comparable visual and data outputs.
"""

__version__ = "0.2.0"  # bumped minor version

# ---- Public configs & types ----
from .types import (
    CfgCommon, CfgHDR, CfgPF, CfgMorph,
    Plane, Variant, ShapeProduct, RunResult,
)


# ---- Main entrypoint ----
from .main_run import align_points, mpase as _mpase

def run(*args, **kwargs):
    """
    Run the MPASE pipeline (recommended user-facing entrypoint).

    This is a convenience alias for the internal `mpase(...)` function.
    """
    return _mpase(*args, **kwargs)

# Backward compatibility / explicit names
mpase = _mpase
run_silhouettes = _mpase  # optional alias

# ---- Visualization (user-facing) ----
from .visualization_save_image import (
    # overlay (multi-condition)
    view, save_figures, view_projections, save_projections,
    # single-condition (new)
    view_single, save_per_label,
)

# ---- Data export (user-facing) ----
from .export_data_for_visd3three import (
    export_aligned_points,
    export_all,
)

# ---- Optional public helper ----
from .io_load import load_points

__all__ = [
    "__version__",
    # main
    "mpase", "run_silhouettes", "align_points",
    # configs / types
    "CfgCommon", "CfgHDR", "CfgPF", "CfgMorph",
    "Plane", "Variant", "ShapeProduct", "RunResult",
    # viz (overlay + single)
    "view", "save_figures", "view_projections", "save_projections",
    "view_single", "save_per_label",
    # exporters
    "export_all", "export_aligned_points",
    # helper
    "load_points",
]
