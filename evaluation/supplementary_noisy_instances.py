"""
Generate side-by-side supplementary figures for reviewer-facing noisy instances.

This script does not modify or extend the MPASE package. It uses unchanged
MPASE outputs for aligned projections and shape masks, and computes the raw
centered point clouds directly from the source CSV files.

Outputs for one real chromosome example:
  - raw_projection_YZ.png: two side-by-side centered point clouds before PCA/ICP alignment
  - projection_YZ.png: two side-by-side MPASE-aligned point clouds
  - hdr_YZ_{100,95,60}.png: two side-by-side HDR shape panels
  - point_fraction_YZ_{100,95,60}.png: two side-by-side PF shape panels

Usage:
    python evaluation/supplementary_noisy_instances.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mpase
from mpase.metrics_calculation import all_contours_from_bool


DATA_ROOT = ROOT / "evaluation" / "data" / "all_structure_files"
OUT_DIR = ROOT / "evaluation" / "supplementary_figures" / "output"
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
DEFAULT_CHROM = "chr1"
EXAMPLES = (("12hrs", "untr"), ("12hrs", "vacv"))
LEVELS = (100, 95, 60)
PLANE_AXES = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
COLORS = ("#1f77b4", "#d62728")

# These match the smoother settings used in examples/example.ipynb to avoid
# tiny HDR islands and PF speckles in publication figures.
CFG_HDR = mpase.CfgHDR(
    n_boot=256,
    sigma_px=1.6,
    density_floor_frac=0.003,
    mass_levels=(1.00, 0.95, 0.60),
)
CFG_PF = mpase.CfgPF(
    frac_levels=(1.00, 0.95, 0.60),
    morph=mpase.CfgMorph(closing=2, opening=2, keep_largest=True, fill_holes=True),
)
CLEAN_BLOBS = True
BLOB_MIN_LEN = 25
BLOB_MIN_AREA_FRAC = 0.01


def _chrom_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.name)
    return (int(match.group(0)) if match else 10_000, path.name)


def _structure_path(chrom: str, hrs: str, cond: str) -> Path:
    return DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"


def _has_example(chrom: str, examples: Iterable[tuple[str, str]]) -> bool:
    return all(_structure_path(chrom, hrs, cond).exists() for hrs, cond in examples)


def choose_chromosome(preferred: str = DEFAULT_CHROM) -> str:
    if _has_example(preferred, EXAMPLES):
        return preferred

    for chrom_dir in sorted((p for p in DATA_ROOT.iterdir() if p.is_dir()), key=_chrom_key):
        if _has_example(chrom_dir.name, EXAMPLES):
            return chrom_dir.name

    raise FileNotFoundError(
        "No chromosome has complete data for "
        + ", ".join(f"{hrs}/{cond}" for hrs, cond in EXAMPLES)
    )


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _pretty_label(hrs: str, cond: str) -> str:
    return f"{cond.upper()} {hrs.replace('hrs', 'h')}"


def collect_inputs(chrom: str) -> tuple[list[str], list[str], list[str]]:
    csvs, labels, pretty = [], [], []
    for hrs, cond in EXAMPLES:
        csvs.append(str(_structure_path(chrom, hrs, cond)))
        labels.append(f"{chrom}_{hrs}_{cond}")
        pretty.append(_pretty_label(hrs, cond))
    return csvs, labels, pretty


def load_centered_points(csvs: list[str]) -> list[np.ndarray]:
    centered = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        pts = df[list(XYZ_COLS)].dropna().values.astype(np.float32)
        centered.append(pts - pts.mean(axis=0))
    return centered


def run_example(chrom: str, plane: str) -> tuple[dict, list[str], list[np.ndarray], list[str]]:
    csvs, labels, pretty = collect_inputs(chrom)
    result = mpase.run(
        csv_list=csvs,
        labels=labels,
        xyz_cols=XYZ_COLS,
        id_col="gene_name",
        cfg_common=mpase.CfgCommon(),
        cfg_hdr=CFG_HDR,
        cfg_pf=CFG_PF,
        planes=(plane,),
    )
    return result, pretty, load_centered_points(csvs), csvs


def _set_shared_2d_limits(axes, point_sets: list[np.ndarray]) -> None:
    stacked = np.vstack(point_sets)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-8)
    pad = radius * 0.08
    for ax in axes:
        ax.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
        ax.set_ylim(center[1] - radius - pad, center[1] + radius + pad)


def save_point_pair(
    point_sets2d: list[np.ndarray],
    pretty: list[str],
    plane: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharex=True, sharey=True)
    for ax, pts, label, color in zip(axes, point_sets2d, pretty, COLORS):
        ax.scatter(pts[:, 0], pts[:, 1], s=4.0, alpha=0.65, color=color)
        ax.set_title(label)
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_aspect("equal")
    _set_shared_2d_limits(axes, point_sets2d)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_projection_csvs(point_sets2d: list[np.ndarray], labels: list[str], plane: str, out_dir: Path, prefix: str) -> None:
    for pts, label in zip(point_sets2d, labels):
        out = out_dir / f"{prefix}_{plane}_{_safe_name(label)}.csv"
        pd.DataFrame(pts, columns=[plane[0], plane[1]]).to_csv(out, index=False)


def _shape_for(result: dict, kind: str, plane: str, level: int, label: str):
    return result.get("shapes", {}).get(kind, {}).get(plane, {}).get(level, {}).get(label)


def _background_for(result: dict, plane: str, label: str):
    if "background_by_label" in result and plane in result["background_by_label"]:
        bg_single = result["background_by_label"][plane].get(label)
        if bg_single is not None:
            return bg_single
    return result.get("background", {}).get(plane)


def plot_single_shape(ax, shape: dict, bg_single: np.ndarray, title: str, color: str) -> None:
    if bg_single is not None:
        ax.imshow(bg_single, cmap="gray", alpha=0.18)

    contours = all_contours_from_bool(
        shape["mask"],
        min_len=BLOB_MIN_LEN,
        min_area_frac=BLOB_MIN_AREA_FRAC if CLEAN_BLOBS else 0.0,
    )
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], "-", lw=2.4, color=color, alpha=0.95)

    ax.set_title(title)
    ax.set_axis_off()


def save_shape_pair(result: dict, pretty: list[str], plane: str, kind: str, level: int, out_dir: Path, dpi: int) -> None:
    labels = result["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    for ax, label, display, color in zip(axes, labels, pretty, COLORS):
        shape = _shape_for(result, kind, plane, level, label)
        if shape is None:
            ax.text(0.5, 0.5, f"No {kind} {level}% for {label}", ha="center", va="center")
            ax.axis("off")
            continue
        plot_single_shape(ax, shape, _background_for(result, plane, label), display, color)

    fig.suptitle(f"{plane} - {kind} {level}%")
    fig.savefig(out_dir / f"{kind}_{plane}_{level}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_point_tables(result: dict, raw_centered: list[np.ndarray], out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    ids_by_label = result.get("ids_by_label", {})

    for key, arrays in (
        ("raw_centered", raw_centered),
        ("aligned", result.get("aligned_points", [])),
    ):
        for label, points in zip(labels, arrays):
            df = pd.DataFrame(np.asarray(points), columns=["x", "y", "z"])
            ids = ids_by_label.get(label)
            if ids is not None and len(ids) == len(df):
                df.insert(0, "gene_id", ids)
            df.to_csv(data_dir / f"{_safe_name(label)}_{key}.csv", index=False)


def save_images(result: dict, pretty: list[str], raw_centered: list[np.ndarray], plane: str, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    i, j = PLANE_AXES[plane]

    raw2d = [pts[:, [i, j]] for pts in raw_centered]
    save_point_pair(
        raw2d,
        pretty,
        plane,
        f"{plane} projection (centered before alignment)",
        out_dir / f"raw_projection_{plane}.png",
        dpi,
    )
    save_projection_csvs(raw2d, labels, plane, out_dir, "raw_projection")

    aligned2d = [result["projections"][plane]["sets"][label] for label in labels]
    save_point_pair(
        aligned2d,
        pretty,
        plane,
        f"{plane} projection (aligned & scaled)",
        out_dir / f"projection_{plane}.png",
        dpi,
    )
    save_projection_csvs(aligned2d, labels, plane, out_dir, "projection")

    for level in LEVELS:
        save_shape_pair(result, pretty, plane, "hdr", level, out_dir, dpi)
        save_shape_pair(result, pretty, plane, "point_fraction", level, out_dir, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", default=DEFAULT_CHROM, help="Preferred chromosome, default: chr1")
    parser.add_argument("--plane", default="YZ", choices=tuple(PLANE_AXES), help="Projection plane")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--dpi", default=400, type=int, help="PNG export DPI")
    args = parser.parse_args()

    chrom = choose_chromosome(args.chrom)
    result, pretty, raw_centered, _ = run_example(chrom, args.plane)

    out_dir = Path(args.out_dir)
    save_images(result, pretty, raw_centered, args.plane, out_dir, args.dpi)
    export_point_tables(result, raw_centered, out_dir)

    print(f"Chromosome: {chrom}")
    print("Examples: " + ", ".join(result["labels"]))
    print(f"Projection plane: {args.plane}")
    print(f"Saved supplementary images: {out_dir}")
    print(f"Saved point tables: {out_dir / 'data'}")


if __name__ == "__main__":
    main()
