#!/usr/bin/env python3
"""
Synthetic-size MPASE runtime scaling benchmarks.

These experiments are computational stress tests only. They resample existing
chromosome point clouds to create larger inputs with fixed random seeds, then
time the unchanged MPASE stages.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mpase")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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
PLANES = ("XY", "YZ", "XZ")
POINT_COUNTS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
SAMPLE_COUNTS = (2, 4, 6, 8, 10, 12, 16, 20)
FIXED_POINTS_PER_SAMPLE = 10_000

CFG_COMMON = mpase.CfgCommon(icp_iters=30)
CFG_HDR = mpase.CfgHDR(
    n_boot=256,
    sigma_px=1.8,
    density_floor_frac=0.002,
    mass_levels=(1.00, 0.95, 0.60),
)
CFG_PF = mpase.CfgPF(
    frac_levels=(1.00, 0.95, 0.60),
    morph=mpase.CfgMorph(
        closing=2,
        opening=2,
        keep_largest=True,
        fill_holes=True,
    ),
)

STAGE_COLUMNS = (
    ("data_preparation", "data_preparation_seconds"),
    ("pca_based_initialization", "pca_initialization_seconds"),
    ("icp_refinement", "icp_refinement_seconds"),
    ("projection_xy_yz_xz", "projection_seconds"),
    ("hdr_shape_extraction", "hdr_shape_extraction_seconds"),
    ("point_fraction_shape_extraction", "point_fraction_shape_extraction_seconds"),
    ("shape_metric_computation", "shape_metric_computation_seconds"),
    ("total_runtime", "total_runtime_seconds"),
)


def _structure_path(chrom: str, hrs: str, cond: str) -> Path:
    return DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"


def load_base_clouds(chrom: str = "chr1") -> list[np.ndarray]:
    bases = []
    for hrs in ("12hrs", "18hrs", "24hrs"):
        for cond in ("untr", "vacv"):
            path = _structure_path(chrom, hrs, cond)
            df = pd.read_csv(path)
            bases.append(df[list(XYZ_COLS)].dropna().values.astype(np.float32))
    if len(bases) < 2:
        raise RuntimeError(f"Need at least two base point clouds under {DATA_ROOT}")
    return bases


def _rotation_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = rng.uniform(-0.20, 0.20)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def prepare_clouds(
    bases: list[np.ndarray],
    *,
    num_samples: int,
    points_per_sample: int,
    seed: int,
) -> tuple[list[np.ndarray], list[str], list[int]]:
    rng = np.random.default_rng(seed)
    clouds = []
    labels = []
    counts = []

    for idx in range(num_samples):
        base = bases[idx % len(bases)]
        choose = rng.integers(0, len(base), size=points_per_sample)
        pts = base[choose].astype(np.float32, copy=True)

        # Small deterministic perturbations avoid exact duplicate samples while
        # keeping the benchmark tied to empirical chromosome coordinate ranges.
        scale = np.std(base, axis=0).astype(np.float32)
        noise = rng.normal(0.0, np.maximum(scale, 1e-6) * 0.01, size=pts.shape)
        pts = pts + noise.astype(np.float32)
        pts = pts @ _rotation_matrix(seed + idx + 1).T
        pts = pts - pts.mean(0)

        clouds.append(pts.astype(np.float32, copy=False))
        labels.append(f"S{idx + 1}")
        counts.append(len(pts))

    return clouds, labels, counts


def align_timed(raw_centered: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
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
            iters=CFG_COMMON.icp_iters,
            sample=CFG_COMMON.sample_icp,
            trim_q=CFG_COMMON.trim_q,
        )
        aligned.append(prealigned @ icp_rot.T + icp_shift)
        icp_seconds += time.perf_counter() - t0

    return aligned, pca_seconds, icp_seconds


def project_timed(
    aligned: list[np.ndarray],
    labels: list[str],
) -> tuple[list[np.ndarray], dict[str, dict[str, Any]], float]:
    t0 = time.perf_counter()
    stacked = np.vstack(aligned)
    mins = stacked.min(0)
    maxs = stacked.max(0)
    scale = float((maxs - mins).max() + 1e-8)
    scaled = [pts / scale for pts in aligned]

    edges3d, _ = make_grid_from_bounds(
        np.vstack(scaled),
        base=CFG_COMMON.grid_base,
        pad_frac=CFG_COMMON.pad_frac,
    )
    projections, _, _ = _per_plane_sets(scaled, edges3d, labels, PLANES)
    return scaled, projections, time.perf_counter() - t0


def hdr_shapes_timed(
    aligned: list[np.ndarray],
    labels: list[str],
) -> tuple[dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]], float]:
    t0 = time.perf_counter()
    edges3d, _ = make_grid_from_bounds(
        np.vstack(aligned),
        base=CFG_COMMON.grid_base,
        pad_frac=CFG_COMMON.pad_frac,
    )

    densities: dict[str, dict[str, np.ndarray]] = {lab: {} for lab in labels}
    for lab, pts in zip(labels, aligned):
        boot = boot_density_2d(
            pts,
            edges3d,
            n_boot=CFG_HDR.n_boot,
            sample_frac=CFG_HDR.sample_frac,
            sigma_px=CFG_HDR.sigma_px,
            rng_seed=CFG_HDR.rng_seed,
        )
        for axis in ("x", "y", "z"):
            plane = PLANE_FROM_AXIS[axis]
            densities[lab][plane] = boot[axis]

    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {"hdr": {}}
    for plane in PLANES:
        shapes["hdr"].setdefault(plane, {})
        for mass in CFG_HDR.mass_levels:
            level = int(round(mass * 100))
            shapes["hdr"][plane].setdefault(level, {})
            for lab in labels:
                shapes["hdr"][plane][level][lab] = make_hdr_shape(
                    densities[lab][plane],
                    plane,
                    mass,
                    CFG_HDR.density_floor_frac,
                )
    return shapes, time.perf_counter() - t0


def pf_shapes_timed(
    projections: dict[str, dict[str, Any]],
    labels: list[str],
) -> tuple[dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]], float]:
    t0 = time.perf_counter()
    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {
        "point_fraction": {}
    }
    for plane, proj in projections.items():
        xs = proj["xs"]
        ys = proj["ys"]
        sets2d = proj["sets"]
        pf_rng = np.random.default_rng(CFG_PF.rng_seed)

        for frac in CFG_PF.frac_levels:
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
                    CFG_PF.bandwidth,
                    CFG_PF.disk_px,
                    morph=CFG_PF.morph,
                    rng=pf_rng,
                )
    return shapes, time.perf_counter() - t0


def metrics_timed(
    shapes: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]],
    labels: list[str],
) -> tuple[int, float]:
    t0 = time.perf_counter()
    metric_rows = 0
    for variant, levels in (
        ("hdr", CFG_HDR.mass_levels),
        ("point_fraction", CFG_PF.frac_levels),
    ):
        for plane in PLANES:
            for level_float in levels:
                level = int(round(level_float * 100))
                level_shapes = shapes[variant][plane][level]
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        a_sp = level_shapes[labels[i]]
                        b_sp = level_shapes[labels[j]]
                        _ = iou_bool(a_sp["mask"], b_sp["mask"])
                        _mean_nn, _hausdorff = contour_distances(
                            a_sp["contour"], b_sp["contour"]
                        )
                        metric_rows += 1
    return metric_rows, time.perf_counter() - t0


def run_condition(
    *,
    experiment: str,
    point_count: int,
    sample_count: int,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    bases = load_base_clouds()
    raw_centered, labels, point_counts = prepare_clouds(
        bases,
        num_samples=sample_count,
        points_per_sample=point_count,
        seed=seed,
    )
    data_preparation_seconds = time.perf_counter() - t0

    aligned_unscaled, pca_seconds, icp_seconds = align_timed(raw_centered)
    aligned, projections, projection_seconds = project_timed(aligned_unscaled, labels)
    hdr_shapes, hdr_seconds = hdr_shapes_timed(aligned, labels)
    pf_shapes, pf_seconds = pf_shapes_timed(projections, labels)
    shapes = {
        "hdr": hdr_shapes["hdr"],
        "point_fraction": pf_shapes["point_fraction"],
    }
    metric_rows, metric_seconds = metrics_timed(shapes, labels)
    total_seconds = time.perf_counter() - total_start

    return {
        "experiment": experiment,
        "point_count_per_sample": point_count,
        "sample_count": sample_count,
        "repeat": repeat,
        "seed": seed,
        "num_points_total": int(sum(point_counts)),
        "num_pairwise_comparisons": int(sample_count * (sample_count - 1) / 2),
        "num_hdr_shapes": int(sample_count * len(PLANES) * len(CFG_HDR.mass_levels)),
        "num_point_fraction_shapes": int(sample_count * len(PLANES) * len(CFG_PF.frac_levels)),
        "num_metric_rows": int(metric_rows),
        "planes": ";".join(PLANES),
        "hdr_levels": ";".join(str(int(round(x * 100))) for x in CFG_HDR.mass_levels),
        "point_fraction_levels": ";".join(
            str(int(round(x * 100))) for x in CFG_PF.frac_levels
        ),
        "data_preparation_seconds": data_preparation_seconds,
        "pca_initialization_seconds": pca_seconds,
        "icp_refinement_seconds": icp_seconds,
        "projection_seconds": projection_seconds,
        "hdr_shape_extraction_seconds": hdr_seconds,
        "point_fraction_shape_extraction_seconds": pf_seconds,
        "shape_metric_computation_seconds": metric_seconds,
        "total_runtime_seconds": total_seconds,
        "status": "completed",
        "notes": "Empirical resampling from existing chr1 chromosome point clouds; computational benchmark only.",
    }


def _worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        queue.put(("ok", run_condition(**kwargs)))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_with_timeout(kwargs: dict[str, Any], timeout_seconds: float) -> tuple[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(queue, kwargs))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timeout", f"Exceeded {timeout_seconds:.0f} seconds"

    try:
        return queue.get_nowait()
    except Empty:
        if proc.exitcode == 0:
            return "error", "Worker exited without returning a result"
        return "error", f"Worker exit code {proc.exitcode}"


def repeats_for_point_count(point_count: int, args: argparse.Namespace) -> int:
    return args.large_repeats if point_count > args.large_repeat_threshold else args.repeats


def summarize(results: pd.DataFrame, condition_column: str) -> pd.DataFrame:
    rows = []
    completed = results[results["status"] == "completed"].copy()
    for condition_value, group in completed.groupby(condition_column, sort=True):
        for stage, column in STAGE_COLUMNS:
            values = group[column].astype(float)
            rows.append(
                {
                    condition_column: condition_value,
                    "stage": stage,
                    "mean_seconds": values.mean(),
                    "min_seconds": values.min(),
                    "max_seconds": values.max(),
                    "num_repeats": int(values.count()),
                    "mean_total_points": group["num_points_total"].mean(),
                    "mean_sample_count": group["sample_count"].mean(),
                    "mean_point_count_per_sample": group["point_count_per_sample"].mean(),
                    "notes": "Completed repeats only; see results CSV for timeouts or failures.",
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    rows: list[dict[str, Any]],
    results_name: str,
    summary_name: str,
    condition_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(rows)
    summary = summarize(results, condition_column)
    results.to_csv(SCRIPT_DIR / results_name, index=False)
    summary.to_csv(SCRIPT_DIR / summary_name, index=False)
    return results, summary


def print_stage_summary(summary: pd.DataFrame, condition_column: str) -> None:
    if summary.empty:
        print("  No completed runs.")
        return
    latest_condition = summary[condition_column].max()
    latest = summary[summary[condition_column] == latest_condition]
    print(f"  Stage means for largest completed {condition_column}={latest_condition}:")
    for _, row in latest.iterrows():
        print(f"    {row['stage']}: {row['mean_seconds']:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--large-repeats", type=int, default=1)
    parser.add_argument("--large-repeat-threshold", type=int, default=50_000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    point_rows: list[dict[str, Any]] = []
    completed_point_counts: list[int] = []
    print("Experiment 1: point-count scaling")
    for point_count in POINT_COUNTS:
        condition_ok = True
        repeats = repeats_for_point_count(point_count, args)
        for repeat in range(1, repeats + 1):
            kwargs = {
                "experiment": "point_count_scaling",
                "point_count": point_count,
                "sample_count": 2,
                "repeat": repeat,
                "seed": args.seed + point_count + repeat,
            }
            print(f"  point_count={point_count}, repeat={repeat}/{repeats}", flush=True)
            status, payload = run_with_timeout(kwargs, args.timeout_seconds)
            if status == "ok":
                row = payload
                point_rows.append(row)
                print(f"    completed total={row['total_runtime_seconds']:.3f}s", flush=True)
            else:
                point_rows.append(
                    {
                        "experiment": "point_count_scaling",
                        "point_count_per_sample": point_count,
                        "sample_count": 2,
                        "repeat": repeat,
                        "seed": kwargs["seed"],
                        "status": status,
                        "notes": payload,
                    }
                )
                print(f"    stopped: {status} ({payload})", flush=True)
                condition_ok = False
                break
        if condition_ok:
            completed_point_counts.append(point_count)
        else:
            break

    point_results, point_summary = save_outputs(
        point_rows,
        "runtime_point_count_scaling_results.csv",
        "runtime_point_count_scaling_summary.csv",
        "point_count_per_sample",
    )

    sample_rows: list[dict[str, Any]] = []
    completed_sample_counts: list[int] = []
    print("\nExperiment 2: sample-count scaling")
    for sample_count in SAMPLE_COUNTS:
        condition_ok = True
        for repeat in range(1, args.repeats + 1):
            kwargs = {
                "experiment": "sample_count_scaling",
                "point_count": FIXED_POINTS_PER_SAMPLE,
                "sample_count": sample_count,
                "repeat": repeat,
                "seed": args.seed + sample_count * 100 + repeat,
            }
            print(f"  sample_count={sample_count}, repeat={repeat}/{args.repeats}", flush=True)
            status, payload = run_with_timeout(kwargs, args.timeout_seconds)
            if status == "ok":
                row = payload
                sample_rows.append(row)
                print(f"    completed total={row['total_runtime_seconds']:.3f}s", flush=True)
            else:
                sample_rows.append(
                    {
                        "experiment": "sample_count_scaling",
                        "point_count_per_sample": FIXED_POINTS_PER_SAMPLE,
                        "sample_count": sample_count,
                        "repeat": repeat,
                        "seed": kwargs["seed"],
                        "status": status,
                        "notes": payload,
                    }
                )
                print(f"    stopped: {status} ({payload})", flush=True)
                condition_ok = False
                break
        if condition_ok:
            completed_sample_counts.append(sample_count)
        else:
            break

    sample_results, sample_summary = save_outputs(
        sample_rows,
        "runtime_sample_count_scaling_results.csv",
        "runtime_sample_count_scaling_summary.csv",
        "sample_count",
    )

    print("\nSaved CSV files:")
    print(f"  {SCRIPT_DIR / 'runtime_point_count_scaling_results.csv'}")
    print(f"  {SCRIPT_DIR / 'runtime_point_count_scaling_summary.csv'}")
    print(f"  {SCRIPT_DIR / 'runtime_sample_count_scaling_results.csv'}")
    print(f"  {SCRIPT_DIR / 'runtime_sample_count_scaling_summary.csv'}")

    print("\nTerminal summary:")
    print(f"  completed point counts: {completed_point_counts}")
    print(f"  completed sample counts: {completed_sample_counts}")
    print(f"  largest completed point count: {max(completed_point_counts) if completed_point_counts else 'none'}")
    print(f"  largest completed sample count: {max(completed_sample_counts) if completed_sample_counts else 'none'}")
    print_stage_summary(point_summary, "point_count_per_sample")
    print_stage_summary(sample_summary, "sample_count")

    completed_all = pd.concat(
        [
            point_results[point_results["status"] == "completed"],
            sample_results[sample_results["status"] == "completed"],
        ],
        ignore_index=True,
    )
    if not completed_all.empty:
        stage_means = {
            stage: float(completed_all[column].mean())
            for stage, column in STAGE_COLUMNS
            if stage != "total_runtime"
        }
        bottleneck = max(stage_means, key=stage_means.get)
        print(
            f"  main bottleneck by measured mean runtime: "
            f"{bottleneck} ({stage_means[bottleneck]:.3f}s)"
        )


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
