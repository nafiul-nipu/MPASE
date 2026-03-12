import os
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .create_grid_planes import (
    AXPAIR,
    PLANE_FROM_AXIS,
    grid_centers_from_edges,
    make_grid_from_bounds,
    rasterize_points,
)
from .hdr_bootstrap import boot_density_2d, make_hdr_shape
from .metrics_calculation import contour_distances, iou_bool
from .point_alignment import best_pca_prealign, icp_rigid_robust
from .point_fraction import make_pf_shape
from .types import CfgCommon, CfgHDR, CfgPF, Plane, ShapeProduct, Variant


def _empty_metrics() -> pd.DataFrame:
    # Keep one canonical empty table shape so every entrypoint returns the same columns.
    return pd.DataFrame(
        columns=["plane", "mode", "level", "A", "B", "IoU", "meanNN", "Hausdorff"]
    )


def _prepare_inputs(
    csv_list: Optional[Sequence[str]] = None,
    *,
    points_list: Optional[Sequence[np.ndarray]] = None,
    labels: Optional[Sequence[str]] = None,
    xyz_cols: Tuple[str, str, str] = ("middle_x", "middle_y", "middle_z"),
    id_col: Optional[str] = None,
    ids_list: Optional[Sequence[Sequence[Any]]] = None,
) -> Tuple[List[np.ndarray], List[str], List[str], List[List[str]]]:
    """
    Load raw point sets and normalize user-facing metadata.

    Return values stay parallel by index:
    - raw_sets[i]
    - labels_out[i]
    - sources[i]
    - ids_per_set[i]
    """
    ids_per_set: List[List[str]] = []

    if points_list is not None:
        raw_sets = [np.asarray(P, dtype=np.float32) for P in points_list]
        labels_out = list(labels) if labels else [f"S{i}" for i in range(len(raw_sets))]
        sources = labels_out[:]

        if ids_list is not None:
            if len(ids_list) != len(raw_sets):
                raise ValueError("ids_list length must match points_list length")
            for ids, pts in zip(ids_list, raw_sets):
                if len(ids) != len(pts):
                    raise ValueError("Each ids_list[i] length must match points_list[i] rows")
                ids_per_set.append([str(x) for x in ids])
        else:
            # Use stable row indices when the caller does not provide explicit IDs.
            ids_per_set = [[str(i) for i in range(len(pts))] for pts in raw_sets]

    elif csv_list:
        raw_sets = []
        labels_out = list(labels) if labels else []
        sources = list(csv_list)

        for csv_path in csv_list:
            # Keep point rows and IDs tied to the same filtered index after dropna.
            df = pd.read_csv(csv_path)
            pts_df = df[list(xyz_cols)].dropna()
            raw_sets.append(pts_df.values.astype(np.float32))

            if not labels:
                labels_out.append(os.path.splitext(os.path.basename(csv_path))[0])

            if id_col is not None:
                if id_col not in df.columns:
                    raise ValueError(f"{id_col!r} not found in {csv_path}")
                ids_per_set.append(df.loc[pts_df.index, id_col].astype(str).tolist())
            else:
                ids_per_set.append([str(ix) for ix in pts_df.index])
    else:
        raise ValueError("Provide csv_list OR points_list.")

    if len(raw_sets) < 2:
        raise ValueError("Need at least 2 point sets for comparison.")

    if len(labels_out) != len(raw_sets):
        raise ValueError("labels length must match the number of input sets")

    return raw_sets, labels_out, sources, ids_per_set


def _per_plane_sets(
    aligned: List[np.ndarray],
    edges3d,
    labels: List[str],
    planes: Tuple[Plane, ...],
) -> Tuple[
    Dict[Plane, Dict[str, object]],
    Dict[Plane, np.ndarray],
    Dict[Plane, Dict[str, np.ndarray]],
]:
    """
    Build shared 2D projections for each requested plane.

    Every label lands on the same pixel grid within a plane, which is what makes
    overlap and contour-distance comparisons valid.
    """
    per_plane: Dict[Plane, Dict[str, object]] = {}
    background: Dict[Plane, np.ndarray] = {}
    background_by_label: Dict[Plane, Dict[str, np.ndarray]] = {}

    for axis in ("x", "y", "z"):
        plane: Plane = PLANE_FROM_AXIS[axis]  # type: ignore[assignment]
        if plane not in planes:
            continue

        i, j = AXPAIR[axis]
        ex, ey = edges3d[i], edges3d[j]
        xs, ys = grid_centers_from_edges(ex, ey)

        # Keep raw 2D projected coordinates because several downstream steps need them.
        sets2d: Dict[str, np.ndarray] = {lab: pts[:, [i, j]] for lab, pts in zip(labels, aligned)}
        per_plane[plane] = {"xs": xs, "ys": ys, "sets": sets2d}

        bg_union = None
        background_by_label[plane] = {}
        for lab in labels:
            cur = rasterize_points(sets2d[lab], xs, ys, disk_px=2)
            background_by_label[plane][lab] = cur
            bg_union = cur if bg_union is None else (bg_union | cur)
        background[plane] = bg_union  # type: ignore[assignment]

    return per_plane, background, background_by_label


def _align_and_project(
    raw_sets: List[np.ndarray],
    labels: List[str],
    sources: List[str],
    ids_per_set: List[List[str]],
    *,
    cfg_common: CfgCommon,
    align_mode: Literal["auto", "skip"],
    planes: Tuple[Plane, ...],
    id_col: Optional[str],
    note: Optional[str] = None,
) -> dict:
    """
    Build the shared alignment/projection result that both public entrypoints use.

    This is the common base for:
    - full analysis via mpase()
    - alignment-only work via align_points()
    """
    ref = raw_sets[0] - raw_sets[0].mean(0)
    aligned: List[np.ndarray] = []

    for idx, pts in enumerate(raw_sets):
        centered = pts - pts.mean(0)

        if idx == 0 or align_mode == "skip":
            # Use the first set as the reference frame, or skip rotational alignment entirely.
            aligned_pts = centered
        else:
            # Use PCA to get close first, then tighten with robust rigid ICP.
            prealign_rot = best_pca_prealign(centered, ref)
            prealigned = centered @ prealign_rot.T
            icp_rot, icp_shift = icp_rigid_robust(
                ref,
                prealigned,
                iters=cfg_common.icp_iters,
                sample=cfg_common.sample_icp,
                trim_q=cfg_common.trim_q,
            )
            aligned_pts = prealigned @ icp_rot.T + icp_shift

        aligned.append(aligned_pts)

    # Normalize everything by one shared scale so every dataset uses one common grid.
    stacked = np.vstack(aligned)
    mins = stacked.min(0)
    maxs = stacked.max(0)
    scale = float((maxs - mins).max() + 1e-8)
    aligned = [pts / scale for pts in aligned]

    edges3d, _ = make_grid_from_bounds(
        np.vstack(aligned),
        base=cfg_common.grid_base,
        pad_frac=cfg_common.pad_frac,
    )
    projections, background, background_by_label = _per_plane_sets(aligned, edges3d, labels, planes)

    meta = {
        "sources": list(sources),
        "labels": list(labels),
        "cfg_common": asdict(cfg_common),
        "cfg_hdr": None,
        "cfg_pf": None,
        "planes": list(projections.keys()),
        "align_mode": align_mode,
        "id_col": id_col,
    }
    if note is not None:
        meta["note"] = note

    return {
        "labels": list(labels),
        "aligned_points": aligned,
        "shapes": {},
        "metrics": _empty_metrics(),
        "meta": meta,
        "background": background,
        "background_by_label": background_by_label,
        "densities": None,
        "projections": projections,
        "ids_by_label": {lab: ids for lab, ids in zip(labels, ids_per_set)},
    }


def align_points(
    csv_list: Optional[Sequence[str]] = None,
    *,
    points_list: Optional[Sequence[np.ndarray]] = None,
    labels: Optional[Sequence[str]] = None,
    xyz_cols: Tuple[str, str, str] = ("middle_x", "middle_y", "middle_z"),
    id_col: Optional[str] = None,
    ids_list: Optional[Sequence[Sequence[Any]]] = None,
    align_mode: Literal["auto", "skip"] = "auto",
    cfg_common: Optional[CfgCommon] = None,
    planes: Tuple[Plane, ...] = ("XY", "YZ", "XZ"),
) -> dict:
    """
    Align point sets, build shared 2D projections, and stop there.

    This is the explicit public path for users who only want aligned coordinates
    and projection-space products, without silhouette extraction or metrics.
    """
    cfg_common = cfg_common or CfgCommon()

    raw_sets, labels_out, sources, ids_per_set = _prepare_inputs(
        csv_list,
        points_list=points_list,
        labels=labels,
        xyz_cols=xyz_cols,
        id_col=id_col,
        ids_list=ids_list,
    )

    return _align_and_project(
        raw_sets,
        labels_out,
        sources,
        ids_per_set,
        cfg_common=cfg_common,
        align_mode=align_mode,
        planes=planes,
        id_col=id_col,
        note="Alignment-only run (no HDR/PF/metrics).",
    )


def mpase(
    csv_list: Optional[Sequence[str]] = None,
    *,
    points_list: Optional[Sequence[np.ndarray]] = None,
    labels: Optional[Sequence[str]] = None,
    xyz_cols: Tuple[str, str, str] = ("middle_x", "middle_y", "middle_z"),
    id_col: Optional[str] = None,
    ids_list: Optional[Sequence[Sequence[Any]]] = None,
    align_mode: Literal["auto", "skip"] = "auto",
    point_alignment_only: bool = False,
    out_dir: Optional[str] = None,
    run_hdr: bool = True,
    run_pf: bool = True,
    cfg_common: Optional[CfgCommon] = None,
    cfg_hdr: Optional[CfgHDR] = None,
    cfg_pf: Optional[CfgPF] = None,
    planes: Tuple[Plane, ...] = ("XY", "YZ", "XZ"),
) -> dict:
    """
    Run the full MPASE pipeline.

    The pipeline always starts from the same base:
    - load inputs
    - align all sets into one shared 3D frame
    - build shared XY / YZ / XZ projection grids

    Then, unless alignment-only mode is requested, it can also:
    - compute HDR silhouettes
    - compute point-fraction silhouettes
    - compute pairwise comparison metrics
    """
    cfg_common = cfg_common or CfgCommon()
    cfg_hdr = cfg_hdr or CfgHDR()
    cfg_pf = cfg_pf or CfgPF()

    raw_sets, labels_out, sources, ids_per_set = _prepare_inputs(
        csv_list,
        points_list=points_list,
        labels=labels,
        xyz_cols=xyz_cols,
        id_col=id_col,
        ids_list=ids_list,
    )

    base_result = _align_and_project(
        raw_sets,
        labels_out,
        sources,
        ids_per_set,
        cfg_common=cfg_common,
        align_mode=align_mode,
        planes=planes,
        id_col=id_col,
        note="Alignment-only run (no HDR/PF/metrics)." if point_alignment_only or (not run_hdr and not run_pf) else None,
    )

    if point_alignment_only or (not run_hdr and not run_pf):
        # Keep the legacy flag working, but stop the computation here.
        if out_dir:
            # Import locally to avoid a circular dependency at module import time.
            from .export_data_for_visd3three import export_aligned_points, export_meta

            export_aligned_points(base_result, out_dir)
            export_meta(base_result, out_dir)
        return base_result

    aligned = base_result["aligned_points"]
    per_plane = base_result["projections"]

    densities: Optional[Dict[str, Dict[Plane, np.ndarray]]] = None
    if run_hdr:
        densities = {lab: {} for lab in labels_out}

        # Rebuild the shared 3D edges from the aligned points so HDR uses the same grid.
        edges3d, _ = make_grid_from_bounds(
            np.vstack(aligned),
            base=cfg_common.grid_base,
            pad_frac=cfg_common.pad_frac,
        )

        for axis in ("x", "y", "z"):
            plane: Plane = PLANE_FROM_AXIS[axis]  # type: ignore[assignment]
            if plane not in planes:
                continue
            for lab, pts in zip(labels_out, aligned):
                boot = boot_density_2d(
                    pts,
                    edges3d,
                    n_boot=cfg_hdr.n_boot,
                    sample_frac=cfg_hdr.sample_frac,
                    sigma_px=cfg_hdr.sigma_px,
                    rng_seed=cfg_hdr.rng_seed,
                )
                densities[lab][plane] = boot[axis]

    shapes: Dict[Variant, Dict[Plane, Dict[int, Dict[str, ShapeProduct]]]] = {}
    rows: List[dict] = []

    if run_hdr and densities is not None:
        shapes.setdefault("hdr", {})
        for plane in per_plane.keys():
            shapes["hdr"].setdefault(plane, {})
            for mass in cfg_hdr.mass_levels:
                level = int(round(mass * 100))
                shapes["hdr"][plane].setdefault(level, {})

                for lab in labels_out:
                    density = densities[lab][plane]
                    if density is None:
                        continue
                    shapes["hdr"][plane][level][lab] = make_hdr_shape(
                        density,
                        plane,
                        mass,
                        cfg_hdr.density_floor_frac,
                    )

                for i in range(len(labels_out)):
                    for j in range(i + 1, len(labels_out)):
                        A_lab, B_lab = labels_out[i], labels_out[j]
                        A_sp = shapes["hdr"][plane][level].get(A_lab)
                        B_sp = shapes["hdr"][plane][level].get(B_lab)
                        if not A_sp or not B_sp:
                            continue
                        mean_nn, hausdorff = contour_distances(A_sp["contour"], B_sp["contour"])
                        rows.append(
                            {
                                "mode": "hdr",
                                "plane": plane,
                                "level": level,
                                "A": A_lab,
                                "B": B_lab,
                                "IoU": iou_bool(A_sp["mask"], B_sp["mask"]),
                                "meanNN": mean_nn,
                                "Hausdorff": hausdorff,
                            }
                        )

    if run_pf:
        shapes.setdefault("point_fraction", {})
        for plane, proj in per_plane.items():
            xs = proj["xs"]
            ys = proj["ys"]
            sets2d: Dict[str, np.ndarray] = proj["sets"]

            for frac in cfg_pf.frac_levels:
                level = int(round(frac * 100))
                shapes["point_fraction"].setdefault(plane, {})
                shapes["point_fraction"][plane].setdefault(level, {})

                for lab, pts2d in sets2d.items():
                    shapes["point_fraction"][plane][level][lab] = make_pf_shape(
                        pts2d,
                        xs,
                        ys,
                        plane,
                        frac,
                        cfg_pf.bandwidth,
                        cfg_pf.disk_px,
                        morph=cfg_pf.morph,
                    )

                for i in range(len(labels_out)):
                    for j in range(i + 1, len(labels_out)):
                        A_lab, B_lab = labels_out[i], labels_out[j]
                        A_sp = shapes["point_fraction"][plane][level].get(A_lab)
                        B_sp = shapes["point_fraction"][plane][level].get(B_lab)
                        if not A_sp or not B_sp:
                            continue
                        mean_nn, hausdorff = contour_distances(A_sp["contour"], B_sp["contour"])
                        rows.append(
                            {
                                "mode": "point_fraction",
                                "plane": plane,
                                "level": level,
                                "A": A_lab,
                                "B": B_lab,
                                "IoU": iou_bool(A_sp["mask"], B_sp["mask"]),
                                "meanNN": mean_nn,
                                "Hausdorff": hausdorff,
                            }
                        )

    metrics = pd.DataFrame(
        rows,
        columns=["plane", "mode", "level", "A", "B", "IoU", "meanNN", "Hausdorff"],
    )
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["plane", "mode", "level", "A", "B"],
            ascending=[True, True, False, True, True],
        )

    base_result["shapes"] = shapes
    base_result["metrics"] = metrics
    base_result["densities"] = densities
    base_result["meta"] = {
        "sources": list(sources),
        "labels": list(labels_out),
        "cfg_common": asdict(cfg_common),
        "cfg_hdr": asdict(cfg_hdr),
        "cfg_pf": asdict(cfg_pf),
        "planes": list(per_plane.keys()),
        "align_mode": align_mode,
        "id_col": id_col,
    }

    return base_result
