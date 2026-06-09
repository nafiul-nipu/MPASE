#!/usr/bin/env python3
"""
Runtime measurements for MPASE real-data chromosome-level runs.

This script reuses the same data, parameters, and internal MPASE calls used by
the real-data evaluation scripts. It only writes CSV timing outputs.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mpase")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import mpase
from mpase.create_grid_planes import PLANE_FROM_AXIS, make_grid_from_bounds
from mpase.hdr_bootstrap import boot_density_2d, make_hdr_shape
from mpase.main_run import _per_plane_sets
from mpase.metrics_calculation import contour_distances, iou_bool
from mpase.point_alignment import best_pca_prealign, icp_rigid_robust
from mpase.point_fraction import make_pf_shape


DATA_ROOT = ROOT / "evaluation" / "data" / "all_structure_files"
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
TIMES = ("12hrs", "18hrs", "24hrs")
CONDS = ("untr", "vacv")
PLANES = ("XY", "YZ", "XZ")
PAPER_CHROMS = (
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr10",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chr23",
    "chr25",
    "chr26",
    "chr27",
    "chr28",
    "chr29",
)
CFG_COMMON = mpase.CfgCommon(icp_iters=30)


class RunSpec(NamedTuple):
    protocol: str
    chromosome: str
    conditions: tuple[tuple[str, str], ...]
    planes: tuple[str, ...]
    cfg_hdr: mpase.CfgHDR
    cfg_pf: mpase.CfgPF


SUPP_CFG_HDR = mpase.CfgHDR(
    n_boot=256,
    sigma_px=1.8,
    density_floor_frac=0.002,
    mass_levels=(1.00, 0.95, 0.60),
)
SUPP_CFG_PF = mpase.CfgPF(
    frac_levels=(1.00, 0.95, 0.60),
    morph=mpase.CfgMorph(
        closing=2,
        opening=2,
        keep_largest=True,
        fill_holes=True,
    ),
)

ALL_CHROMOSOME_RUNS = tuple(
    RunSpec(
        protocol="all_chromosome_supplementary_params",
        chromosome=chrom,
        conditions=tuple((hrs, cond) for hrs in TIMES for cond in CONDS),
        planes=PLANES,
        cfg_hdr=SUPP_CFG_HDR,
        cfg_pf=SUPP_CFG_PF,
    )
    for chrom in PAPER_CHROMS
)


def _gene_info(chrom: str, hrs: str, cond: str) -> Path:
    return DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"


def collect_spec_inputs(spec: RunSpec) -> tuple[list[Path], list[str]]:
    csvs: list[Path] = []
    labels: list[str] = []
    for hrs, cond in spec.conditions:
        csv_path = _gene_info(spec.chromosome, hrs, cond)
        if csv_path.exists():
            csvs.append(csv_path)
            labels.append(f"{spec.chromosome}_{hrs}_{cond}")
    return csvs, labels


def load_and_preprocess(csvs: list[Path]) -> tuple[list[np.ndarray], list[int]]:
    raw_sets: list[np.ndarray] = []
    point_counts: list[int] = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        pts_df = df[list(XYZ_COLS)].dropna()
        pts = pts_df.values.astype(np.float32)
        raw_sets.append(pts - pts.mean(0))
        point_counts.append(len(pts))
    return raw_sets, point_counts


def align_timed(
    raw_centered: list[np.ndarray],
    cfg_common: mpase.CfgCommon,
) -> tuple[list[np.ndarray], float, float]:
    ref = raw_centered[0]
    aligned = [ref]
    pca_seconds = 0.0
    icp_seconds = 0.0

    for centered in raw_centered[1:]:
        t0 = time.perf_counter()
        prealign_rot = best_pca_prealign(centered, ref)
        prealigned = centered @ prealign_rot.T
        pca_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        icp_rot, icp_shift = icp_rigid_robust(
            ref,
            prealigned,
            iters=cfg_common.icp_iters,
            sample=cfg_common.sample_icp,
            trim_q=cfg_common.trim_q,
        )
        aligned.append(prealigned @ icp_rot.T + icp_shift)
        icp_seconds += time.perf_counter() - t0

    return aligned, pca_seconds, icp_seconds


def project_timed(
    aligned: list[np.ndarray],
    labels: list[str],
    cfg_common: mpase.CfgCommon,
    planes: tuple[str, ...],
) -> tuple[list[np.ndarray], dict[str, dict[str, Any]], float]:
    t0 = time.perf_counter()
    stacked = np.vstack(aligned)
    mins = stacked.min(0)
    maxs = stacked.max(0)
    scale = float((maxs - mins).max() + 1e-8)
    scaled = [pts / scale for pts in aligned]

    edges3d, _ = make_grid_from_bounds(
        np.vstack(scaled),
        base=cfg_common.grid_base,
        pad_frac=cfg_common.pad_frac,
    )
    projections, _, _ = _per_plane_sets(scaled, edges3d, labels, planes)
    return scaled, projections, time.perf_counter() - t0


def hdr_shapes_timed(
    aligned: list[np.ndarray],
    labels: list[str],
    cfg_common: mpase.CfgCommon,
    cfg_hdr: mpase.CfgHDR,
    planes: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]], float]:
    t0 = time.perf_counter()
    edges3d, _ = make_grid_from_bounds(
        np.vstack(aligned),
        base=cfg_common.grid_base,
        pad_frac=cfg_common.pad_frac,
    )

    densities: dict[str, dict[str, np.ndarray]] = {lab: {} for lab in labels}
    for lab, pts in zip(labels, aligned):
        boot = boot_density_2d(
            pts,
            edges3d,
            n_boot=cfg_hdr.n_boot,
            sample_frac=cfg_hdr.sample_frac,
            sigma_px=cfg_hdr.sigma_px,
            rng_seed=cfg_hdr.rng_seed,
        )
        for axis in ("x", "y", "z"):
            plane = PLANE_FROM_AXIS[axis]
            if plane in planes:
                densities[lab][plane] = boot[axis]

    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {"hdr": {}}
    for plane in planes:
        shapes["hdr"].setdefault(plane, {})
        for mass in cfg_hdr.mass_levels:
            level = int(round(mass * 100))
            shapes["hdr"][plane].setdefault(level, {})
            for lab in labels:
                shapes["hdr"][plane][level][lab] = make_hdr_shape(
                    densities[lab][plane],
                    plane,
                    mass,
                    cfg_hdr.density_floor_frac,
                )
    return shapes, time.perf_counter() - t0


def pf_shapes_timed(
    projections: dict[str, dict[str, Any]],
    labels: list[str],
    cfg_pf: mpase.CfgPF,
) -> tuple[dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]], float]:
    t0 = time.perf_counter()
    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {
        "point_fraction": {}
    }
    for plane, proj in projections.items():
        xs = proj["xs"]
        ys = proj["ys"]
        sets2d = proj["sets"]
        pf_rng = np.random.default_rng(cfg_pf.rng_seed)

        for frac in cfg_pf.frac_levels:
            level = int(round(frac * 100))
            shapes["point_fraction"].setdefault(plane, {})
            shapes["point_fraction"][plane].setdefault(level, {})
            for lab in labels:
                shapes["point_fraction"][plane][level][lab] = make_pf_shape(
                    sets2d[lab],
                    xs,
                    ys,
                    plane,
                    frac,
                    cfg_pf.bandwidth,
                    cfg_pf.disk_px,
                    morph=cfg_pf.morph,
                    rng=pf_rng,
                )
    return shapes, time.perf_counter() - t0


def metrics_timed(
    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]],
    labels: list[str],
    cfg_hdr: mpase.CfgHDR,
    cfg_pf: mpase.CfgPF,
    planes: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    rows = []
    for variant, cfg_levels in (
        ("hdr", cfg_hdr.mass_levels),
        ("point_fraction", cfg_pf.frac_levels),
    ):
        for plane in planes:
            for level_float in cfg_levels:
                level = int(round(level_float * 100))
                level_shapes = shapes[variant][plane][level]
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        a_lab = labels[i]
                        b_lab = labels[j]
                        a_sp = level_shapes[a_lab]
                        b_sp = level_shapes[b_lab]
                        mean_nn, hausdorff = contour_distances(
                            a_sp["contour"], b_sp["contour"]
                        )
                        rows.append(
                            {
                                "plane": plane,
                                "variant": variant,
                                "level": level,
                                "A": a_lab,
                                "B": b_lab,
                                "IoU": iou_bool(a_sp["mask"], b_sp["mask"]),
                                "meanNN": mean_nn,
                                "Hausdorff": hausdorff,
                            }
                        )
    metrics = pd.DataFrame(
        rows,
        columns=["plane", "variant", "level", "A", "B", "IoU", "meanNN", "Hausdorff"],
    )
    return metrics, time.perf_counter() - t0


def time_spec(spec: RunSpec) -> dict[str, Any] | None:
    csvs, labels = collect_spec_inputs(spec)
    if len(csvs) != len(spec.conditions):
        print(
            f"{spec.protocol} {spec.chromosome}: skipping, "
            f"found {len(csvs)} of {len(spec.conditions)} expected files",
            flush=True,
        )
        return None

    print(
        f"{spec.protocol} {spec.chromosome}: timing "
        f"{len(spec.conditions)} input file(s), planes={','.join(spec.planes)}",
        flush=True,
    )
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    raw_centered, point_counts = load_and_preprocess(csvs)
    loading_seconds = time.perf_counter() - t0

    aligned_unscaled, pca_seconds, icp_seconds = align_timed(raw_centered, CFG_COMMON)
    aligned, projections, projection_seconds = project_timed(
        aligned_unscaled, labels, CFG_COMMON, spec.planes
    )
    hdr_shapes, hdr_seconds = hdr_shapes_timed(
        aligned, labels, CFG_COMMON, spec.cfg_hdr, spec.planes
    )
    pf_shapes, pf_seconds = pf_shapes_timed(projections, labels, spec.cfg_pf)

    shapes = {
        "hdr": hdr_shapes["hdr"],
        "point_fraction": pf_shapes["point_fraction"],
    }
    metrics, metric_seconds = metrics_timed(
        shapes, labels, spec.cfg_hdr, spec.cfg_pf, spec.planes
    )

    total_seconds = time.perf_counter() - total_start
    total_after_3d_seconds = (
        projection_seconds + hdr_seconds + pf_seconds + metric_seconds
    )

    condition_time = ";".join(f"{hrs}_{cond}" for hrs, cond in spec.conditions)
    row = {
        "protocol": spec.protocol,
        "chromosome": spec.chromosome,
        "condition_time_comparison": condition_time,
        "planes": ";".join(spec.planes),
        "levels_hdr": ";".join(str(int(round(x * 100))) for x in spec.cfg_hdr.mass_levels),
        "levels_point_fraction": ";".join(
            str(int(round(x * 100))) for x in spec.cfg_pf.frac_levels
        ),
        "labels": ";".join(labels),
        "num_input_files": len(csvs),
        "num_points_total": int(sum(point_counts)),
        "num_points_min": int(min(point_counts)),
        "num_points_max": int(max(point_counts)),
        "num_genes": int(sum(point_counts)),
        "num_pairwise_comparisons": int(len(labels) * (len(labels) - 1) / 2),
        "num_metric_rows": int(len(metrics)),
        "data_loading_preprocessing_seconds": loading_seconds,
        "pca_initialization_seconds": pca_seconds,
        "icp_refinement_seconds": icp_seconds,
        "projection_seconds": projection_seconds,
        "hdr_shape_extraction_seconds": hdr_seconds,
        "point_fraction_shape_extraction_seconds": pf_seconds,
        "shape_metric_computation_seconds": metric_seconds,
        "total_after_3d_coordinates_available_seconds": total_after_3d_seconds,
        "total_runtime_seconds": total_seconds,
    }
    print(
        f"{spec.protocol} {spec.chromosome}: total={total_seconds:.3f}s, "
        f"after_3d={total_after_3d_seconds:.3f}s",
        flush=True,
    )
    return row


def build_summary(results: pd.DataFrame, notes: str) -> pd.DataFrame:
    stage_columns = [
        (
            "data_loading_preprocessing",
            "data_loading_preprocessing_seconds",
            "CSV read, coordinate extraction, dropna, centering.",
        ),
        (
            "pca_based_initialization",
            "pca_initialization_seconds",
            "best_pca_prealign plus applying the prealignment rotation.",
        ),
        (
            "icp_refinement",
            "icp_refinement_seconds",
            "icp_rigid_robust with paper config.",
        ),
        (
            "projection_xy_yz_xz",
            "projection_seconds",
            "Shared scaling, grid construction, and XY/YZ/XZ projections.",
        ),
        (
            "hdr_shape_extraction",
            "hdr_shape_extraction_seconds",
            "Bootstrap density maps and HDR masks/contours.",
        ),
        (
            "point_fraction_shape_extraction",
            "point_fraction_shape_extraction_seconds",
            "KDE-ranked point-fraction masks/contours.",
        ),
        (
            "shape_metric_computation",
            "shape_metric_computation_seconds",
            "IoU, mean nearest-neighbor distance, and Hausdorff over all metric rows.",
        ),
        (
            "total_after_3d_coordinates_available",
            "total_after_3d_coordinates_available_seconds",
            "Projection + HDR + point-fraction + metrics after alignment output exists.",
        ),
        (
            "total_runtime",
            "total_runtime_seconds",
            "End-to-end timed run for the chromosome-level real-data inputs.",
        ),
    ]

    rows = []
    point_totals = results["num_points_total"].astype(float)
    point_mins = results["num_points_min"].astype(float)
    point_maxs = results["num_points_max"].astype(float)
    for stage, column, stage_note in stage_columns:
        values = results[column].astype(float)
        rows.append(
            {
                "stage": stage,
                "mean_seconds": values.mean(),
                "std_seconds": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min_seconds": values.min(),
                "max_seconds": values.max(),
                "num_runs": int(values.count()),
                "mean_total_points": point_totals.mean(),
                "min_total_points": point_totals.min(),
                "max_total_points": point_totals.max(),
                "mean_min_points_per_input_file": point_mins.mean(),
                "mean_max_points_per_input_file": point_maxs.mean(),
                "notes": f"{stage_note} {notes}",
            }
        )
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    first = summary.iloc[0]
    print(
        "\nPoint counts per chromosome-level run: "
        f"mean_total={first['mean_total_points']:.0f}, "
        f"min_total={first['min_total_points']:.0f}, "
        f"max_total={first['max_total_points']:.0f}"
    )
    print("\nRuntime summary (seconds):")
    for _, row in summary.iterrows():
        print(
            f"  {row['stage']}: mean={row['mean_seconds']:.3f}, "
            f"min={row['min_seconds']:.3f}, max={row['max_seconds']:.3f}"
        )

    core = summary[
        ~summary["stage"].isin(
            ["total_after_3d_coordinates_available", "total_runtime"]
        )
    ].copy()
    slowest = core.sort_values("mean_seconds", ascending=False).iloc[0]
    print(
        f"\nSlowest measured stage by mean runtime: "
        f"{slowest['stage']} ({slowest['mean_seconds']:.3f}s)"
    )


def main() -> None:
    out_dir = SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in ALL_CHROMOSOME_RUNS:
        row = time_spec(spec)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No chromosome timing runs completed.")

    results = pd.DataFrame(rows)
    notes = (
        "All available paper chromosome-level six-condition runs using "
        "the supplementary noisy-instance/visual-comparison parameters. "
        f"cfg_common={asdict(CFG_COMMON)}. "
        f"cfg_hdr={asdict(SUPP_CFG_HDR)}, cfg_pf={asdict(SUPP_CFG_PF)}, "
        f"planes={PLANES}, xyz_cols={XYZ_COLS}. "
        "Data loading/preprocessing means reading each CSV, selecting coordinate "
        "columns, dropping rows with missing coordinates, converting to float32, "
        "and centering each point cloud by subtracting its mean XYZ coordinate."
    )
    summary = build_summary(results, notes)

    results_path = out_dir / "runtime_stage_results_actual_data.csv"
    summary_path = out_dir / "runtime_stage_summary_actual_data.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved {results_path}")
    print(f"Saved {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
