import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import re

from skimage.measure import find_contours  
from .metrics_calculation import all_contours_from_bool
from .types import CfgHDR, CfgPF, RunResult, Plane, ShapeProduct


################################################### Visualization and Saving ##########################

# pick two labels to visualize
def _pick_ab_labels(result, A_lab: Optional[str], B_lab: Optional[str]) -> Tuple[str, str]:
    """
    Choose which two result labels to display.
    Defaults to the first two in result['labels'] if not given.
    """
    all_labels = result.get("labels", [])
    if not all_labels or len(all_labels) < 2:
        raise ValueError("Result must contain at least two labels in result['labels'].")
    A = A_lab if A_lab is not None else all_labels[0]
    B = B_lab if B_lab is not None else all_labels[1]
    if A not in all_labels or B not in all_labels:
        raise ValueError(f"A_lab/B_lab must be among result['labels']: {all_labels}")
    if A == B:
        raise ValueError("A_lab and B_lab must refer to two different sets.")
    return A, B


# parse levels argument (int, list[int], or "all") into a tuple of ints
def _levels_from_result(kind: str, cfg_hdr: CfgHDR, cfg_pf: CfgPF, levels):
    # supports int, list[int], or "all"
    if isinstance(levels, str) and levels.lower() == "all":
        if kind == "hdr":
            return tuple(sorted({int(round(p*100)) for p in cfg_hdr.mass_levels}, reverse=True))
        if kind == "point_fraction":
            return tuple(sorted({int(round(p*100)) for p in cfg_pf.frac_levels}, reverse=True))
    if isinstance(levels, (list, tuple, set, np.ndarray)):
        return tuple(int(x) for x in levels)
    if isinstance(levels, (int, float)):
        return (int(levels),)
    raise ValueError("levels must be int, list[int], or 'all'")

# simple overlay plot of two shapes on given axis
def _plot_overlay(
    ax,
    shapeA: ShapeProduct,
    shapeB: ShapeProduct,
    bg: np.ndarray,
    title: str,
    labelA: str = "A",
    labelB: str = "B",
    *,
    clean: bool = False,
    blob_min_len: int = 10,
    blob_min_area_frac: float = 0.05,
):
    if bg is not None:
        ax.imshow(bg, cmap="gray", alpha=0.12)

    # --- A ---
    maskA = shapeA.get("mask")
    if maskA is not None:
        contoursA = all_contours_from_bool(
            maskA,
            min_len=blob_min_len,
            min_area_frac=blob_min_area_frac if clean else 0.0,
        )
        label_str_A = f"{labelA} {shapeA['variant']} {shapeA['level']}%"
        for i, C in enumerate(contoursA):
            ax.plot(
                C[:, 1],
                C[:, 0],
                "-",
                lw=2.4,
                color="#1f77b4",
                alpha=0.95,
                label=label_str_A if i == 0 else None,
            )

    # --- B ---
    maskB = shapeB.get("mask")
    if maskB is not None:
        contoursB = all_contours_from_bool(
            maskB,
            min_len=blob_min_len,
            min_area_frac=blob_min_area_frac if clean else 0.0,
        )
        label_str_B = f"{labelB} {shapeB['variant']} {shapeB['level']}%"
        for i, C in enumerate(contoursB):
            ax.plot(
                C[:, 1],
                C[:, 0],
                "-",
                lw=2.4,
                color="#d62728",
                alpha=0.95,
                label=label_str_B if i == 0 else None,
            )

    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(frameon=False, loc="upper right")



def view(result: RunResult,
         kind: Literal["hdr","point_fraction"] = "hdr",
         plane: Plane = "XY",
         levels: "int|list[int]|str" = "all",
         *,
         # Which two sets to visualize from the N-set result
         A_lab: Optional[str] = None,
         B_lab: Optional[str] = None,
         # Pretty labels for the legend (purely cosmetic)
         labelA: str = "A",
         labelB: str = "B",
         show_heat: bool = False,
         cfg_hdr: Optional[CfgHDR] = None,
         cfg_pf: Optional[CfgPF] = None,
         clean_blobs: bool = False,
         blob_min_len: int = 10,
         blob_min_area_frac: float = 0.05):
    """
    Quick viewer for shapes produced by mpase() [N-set].
    - kind: 'hdr' | 'point_fraction'
    - plane: 'XY' | 'YZ' | 'XZ'
    - levels: int | list[int] | 'all'
    - A_lab/B_lab: which two result labels to show (default = first two in result['labels'])
    - show_heat: if True and kind='hdr' and densities exist, show A_lab's density heat.
    """
    cfg_hdr = cfg_hdr or CfgHDR()
    cfg_pf  = cfg_pf  or CfgPF()
    lvls = _levels_from_result(kind, cfg_hdr, cfg_pf, levels)

    # choose which sets to visualize
    A_key, B_key = _pick_ab_labels(result, A_lab, B_lab)

    if kind not in result["shapes"]:
        print(f"[view] No shapes for kind='{kind}'. Did you run with run_{kind}=True?")
        return
    if plane not in result["shapes"][kind]:
        print(f"[view] No shapes for plane '{plane}' and kind '{kind}'.")
        return

    # dict: level -> {label -> ShapeProduct}
    shapes_by_plane = result["shapes"][kind][plane]

    # figure
    fig, axes = plt.subplots(1, len(lvls), figsize=(5.4*len(lvls), 5.2))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # density heat if available (A_key only, for context)
    D_A_plane = None
    if show_heat and kind == "hdr" and result.get("densities") and result["densities"].get(A_key):
        D_A_plane = result["densities"][A_key].get(plane)

    for ax, lvl in zip(axes, lvls):
        by_label = shapes_by_plane.get(lvl)
        if by_label is None:
            ax.text(0.5, 0.5, f"No {kind} {lvl}%", ha="center", va="center"); ax.axis("off"); continue

        spA = by_label.get(A_key)
        spB = by_label.get(B_key)
        if spA is None or spB is None:
            ax.text(0.5, 0.5, f"Missing {A_key if spA is None else B_key} @ {lvl}%", ha="center", va="center"); ax.axis("off"); continue

        bg = result["background"].get(plane)
        if D_A_plane is not None:
            ax.imshow(D_A_plane, alpha=0.35)  # faint heat underlay
        _plot_overlay(ax, spA, spB, bg, f"{plane} — {kind} {lvl}%", labelA, labelB, clean=clean_blobs,
                      blob_min_len=blob_min_len, blob_min_area_frac=blob_min_area_frac)

    plt.tight_layout(); plt.show()

def save_figures(result: RunResult,
                 kind: Literal["hdr","point_fraction"] = "hdr",
                 plane: Plane = "XY",
                 levels: "int|list[int]|str" = "all",
                 *,
                 out_dir: str = "figures",
                 # Which two sets to visualize from the N-set result
                 A_lab: Optional[str] = None,
                 B_lab: Optional[str] = None,
                 # Pretty labels for the legend (purely cosmetic)
                 labelA: str = "A",
                 labelB: str = "B",
                 show_heat: bool = False,
                 cfg_hdr: Optional[CfgHDR] = None,
                 cfg_pf: Optional[CfgPF] = None,
                 clean_blobs: bool = False,
                 blob_min_len: int = 10,
                 blob_min_area_frac: float = 0.05):
    """
    Save per-plane overlays as PNGs for the selected kind/levels.
    Signature is consistent with `view()`.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_hdr = cfg_hdr or CfgHDR()
    cfg_pf  = cfg_pf  or CfgPF()
    lvls = _levels_from_result(kind, cfg_hdr, cfg_pf, levels)

    # choose which sets to visualize
    A_key, B_key = _pick_ab_labels(result, A_lab, B_lab)

    if kind not in result["shapes"]:
        print(f"[save_figures] No shapes for kind='{kind}'.")
        return
    if plane not in result["shapes"][kind]:
        print(f"[save_figures] No shapes for plane='{plane}' and kind='{kind}'.")
        return

    shapes_by_plane = result["shapes"][kind][plane]

    # density heat if requested (A_key)
    D_A_plane = None
    if show_heat and kind=="hdr" and result.get("densities") and result["densities"].get(A_key):
        D_A_plane = result["densities"][A_key].get(plane)

    for lvl in lvls:
        by_label = shapes_by_plane.get(lvl)
        if by_label is None:
            continue
        spA = by_label.get(A_key)
        spB = by_label.get(B_key)
        if spA is None or spB is None:
            continue
        bg = result["background"].get(plane)

        fig, ax = plt.subplots(figsize=(5.2,5.2))
        if D_A_plane is not None:
            ax.imshow(D_A_plane, alpha=0.35, cmap="inferno")
        _plot_overlay(ax, spA, spB, bg,
                      f"{plane} — {kind} {lvl}%",
                      labelA, labelB, clean=clean_blobs,
                      blob_min_len=blob_min_len, blob_min_area_frac=blob_min_area_frac)

        fname = f"{kind}_{plane}_{lvl}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)


def view_projections(result: RunResult,
                     *,
                     planes: Tuple[Plane, ...] = ("XY","YZ","XZ"),
                     # Which two sets to visualize from the N-set result
                     A_lab: Optional[str] = None,
                     B_lab: Optional[str] = None,
                     # Pretty labels for legend only
                     labelA: str = "A",
                     labelB: str = "B",
                     s: float = 3.0,
                     alphaA: float = 0.7,
                     alphaB: float = 0.7):
    """
    Show aligned point scatter plots for each plane (N-set result; pick any two to display).
    """
    proj = result.get("projections", {})
    if not proj:
        print("[view_projections] No projections in result.")
        return

    # choose which sets to visualize
    A_key, B_key = _pick_ab_labels(result, A_lab, B_lab)

    for plane in planes:
        if plane not in proj:
            print(f"[view_projections] No plane '{plane}' in projections.")
            continue

        sets2d = proj[plane]["sets"]
        if A_key not in sets2d or B_key not in sets2d:
            print(f"[view_projections] Missing {A_key} or {B_key} in projections[{plane}]['sets'].")
            continue

        A2 = sets2d[A_key]; B2 = sets2d[B_key]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(A2[:,0], A2[:,1], s=s, alpha=alphaA, label=labelA)
        ax.scatter(B2[:,0], B2[:,1], s=s, alpha=alphaB, label=labelB)
        ax.set_title(f"{plane} projection (aligned & scaled)")
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
        ax.set_aspect("equal"); ax.legend(frameon=False)
        plt.show()


def save_projections(result: RunResult,
                     *,
                     out_dir: str = "figures",
                     planes: Tuple[Plane, ...] = ("XY","YZ","XZ"),
                     # Which two sets to visualize from the N-set result
                     A_lab: Optional[str] = None,
                     B_lab: Optional[str] = None,
                     # Pretty labels for legend only
                     labelA: str = "A",
                     labelB: str = "B",
                     s: float = 3.0,
                     alphaA: float = 0.7,
                     alphaB: float = 0.7,
                     dpi: int = 220,
                     save_csv: bool = False):
    """
    Save aligned point scatter plots (and optionally CSVs) for each plane (N-set result; pick any two).
    """
    os.makedirs(out_dir, exist_ok=True)
    proj = result.get("projections", {})
    if not proj:
        print("[save_projections] No projections in result.")
        return

    # choose which sets to visualize
    A_key, B_key = _pick_ab_labels(result, A_lab, B_lab)

    for plane in planes:
        if plane not in proj:
            print(f"[save_projections] No plane '{plane}' in projections.")
            continue

        sets2d = proj[plane]["sets"]
        if A_key not in sets2d or B_key not in sets2d:
            print(f"[save_projections] Missing {A_key} or {B_key} in projections[{plane}]['sets'].")
            continue

        A2 = sets2d[A_key]; B2 = sets2d[B_key]

        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(A2[:,0], A2[:,1], s=s, alpha=alphaA, label=labelA)
        ax.scatter(B2[:,0], B2[:,1], s=s, alpha=alphaB, label=labelB)
        ax.set_title(f"{plane} projection (aligned & scaled)")
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
        ax.set_aspect("equal"); ax.legend(frameon=False)
        fig.savefig(os.path.join(out_dir, f"projection_{plane}.png"),
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if save_csv:
            base = os.path.join(out_dir, f"projection_{plane}")
            pd.DataFrame(A2, columns=[f"{plane[0]}", f"{plane[1]}"]).to_csv(base + f"_{A_key}.csv", index=False)
            pd.DataFrame(B2, columns=[f"{plane[0]}", f"{plane[1]}"]).to_csv(base + f"_{B_key}.csv", index=False)

# --------------------------- NEW: single-label (no overlay) ---------------------------

def _plot_single(
    ax,
    shape: ShapeProduct,
    bg_single: np.ndarray,
    title: str,
    color: str = "#1f77b4",
    *,
    clean: bool = False,
    blob_min_len: int = 10,
    blob_min_area_frac: float = 0.05,
):
    if bg_single is not None:
        ax.imshow(bg_single, cmap="gray", alpha=0.18)

    mask = shape.get("mask")
    if mask is not None:
        contours = all_contours_from_bool(
            mask,
            min_len=blob_min_len,
            min_area_frac=blob_min_area_frac if clean else 0.0,
        )
        label_str = f"{shape['variant']} {shape['level']}%"
        for i, C in enumerate(contours):
            ax.plot(
                C[:, 1],
                C[:, 0],
                "-",
                lw=2.4,
                color=color,
                alpha=0.95,
                label=label_str if i == 0 else None,
            )
        if contours:
            ax.legend(frameon=False, loc="upper right")

    ax.set_title(title)
    ax.set_axis_off()




def view_single(result: RunResult,
                label: str,
                kind: Literal["hdr","point_fraction"] = "hdr",
                plane: Plane = "XY",
                levels: "int|list[int]|str" = "all",
                *,
                show_heat: bool = False,
                cfg_hdr: Optional[CfgHDR] = None,
                cfg_pf: Optional[CfgPF] = None, clean_blobs: bool = False,
                blob_min_len: int = 10,
                blob_min_area_frac: float = 0.05):
    """
    View one label at a time (no overlay). If per-label background masks exist, use them.
    """
    cfg_hdr = cfg_hdr or CfgHDR()
    cfg_pf  = cfg_pf  or CfgPF()
    lvls = _levels_from_result(kind, cfg_hdr, cfg_pf, levels)

    if kind not in result.get("shapes", {}):
        print(f"[view_single] No shapes for kind='{kind}'.")
        return
    if plane not in result["shapes"][kind]:
        print(f"[view_single] No shapes for plane='{plane}' and kind='{kind}'.")
        return

    by_level = result["shapes"][kind][plane]
    # background selection
    bg_single = None
    if "background_by_label" in result and plane in result["background_by_label"]:
        bg_single = result["background_by_label"][plane].get(label)
    if bg_single is None:
        bg_single = result.get("background", {}).get(plane)

    # optional density heat for HDR
    D_plane = None
    if show_heat and kind == "hdr" and result.get("densities") and result["densities"].get(label):
        D_plane = result["densities"][label].get(plane)

    fig, axes = plt.subplots(1, len(lvls), figsize=(5.4*len(lvls), 5.2))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, lvl in zip(axes, lvls):
        d = by_level.get(lvl)
        if not d or label not in d:
            ax.text(0.5, 0.5, f"No {kind} {lvl}% for {label}", ha="center", va="center"); ax.axis("off"); continue
        sp = d[label]
        if D_plane is not None:
            ax.imshow(D_plane, alpha=0.35)
        _plot_single(ax, sp, bg_single, f"{label} — {plane} — {kind} {lvl}%", clean=clean_blobs,
                      blob_min_len=blob_min_len, blob_min_area_frac=blob_min_area_frac)

    plt.tight_layout(); plt.show()


def save_per_label(result: RunResult,
                   labels: Optional[Tuple[str, ...]] = None,
                   *,
                   kind: Literal["hdr","point_fraction"] = "hdr",
                   plane: Plane = "XY",
                   levels: "int|list[int]|str" = "all",
                   out_dir: str = "figures_single",
                   show_heat: bool = False,
                   cfg_hdr: Optional[CfgHDR] = None,
                   cfg_pf: Optional[CfgPF] = None,
                   dpi: int = 220, clean_blobs: bool = False,
                   blob_min_len: int = 10,
                   blob_min_area_frac: float = 0.05):
    """
    Save figures for one or more labels without overlay. Each label × level gets its own PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_hdr = cfg_hdr or CfgHDR()
    cfg_pf  = cfg_pf  or CfgPF()
    lvls = _levels_from_result(kind, cfg_hdr, cfg_pf, levels)

    all_labels = list(result.get("labels", []))
    labels = labels or tuple(all_labels)

    if kind not in result.get("shapes", {}):
        print(f"[save_per_label] No shapes for kind='{kind}'.")
        return
    if plane not in result["shapes"][kind]:
        print(f"[save_per_label] No shapes for plane='{plane}' and kind='{kind}'.")
        return

    by_level = result["shapes"][kind][plane]

    for lab in labels:
        # background selection: per-label preferred
        bg_single = None
        if "background_by_label" in result and plane in result["background_by_label"]:
            bg_single = result["background_by_label"][plane].get(lab)
        if bg_single is None:
            bg_single = result.get("background", {}).get(plane)

        D_plane = None
        if show_heat and kind == "hdr" and result.get("densities") and result["densities"].get(lab):
            D_plane = result["densities"][lab].get(plane)

        for lvl in lvls:
            d = by_level.get(lvl)
            if not d or lab not in d:
                continue
            sp = d[lab]
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            if D_plane is not None:
                ax.imshow(D_plane, alpha=0.35, cmap="inferno")
            _plot_single(ax, sp, bg_single, f"{lab} — {plane} — {kind} {lvl}%", clean=clean_blobs,
                          blob_min_len=blob_min_len, blob_min_area_frac=blob_min_area_frac)
            fname = f"{re.sub(r'[^A-Za-z0-9_.-]+','_', lab)}_{kind}_{plane}_{lvl}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
