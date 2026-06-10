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
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mpase
from mpase.create_grid_planes import points_to_pixel_indices
from mpase.metrics_calculation import all_contours_from_bool
from mpase.visualization_save_image import _plot_single as mpase_plot_single


DATA_ROOT = ROOT / "evaluation" / "data" / "all_structure_files"
OUT_DIR = ROOT / "evaluation" / "supplementary_figures" / "output" / "noisy_instances" / "current"
CANDIDATE_DIR = ROOT / "evaluation" / "supplementary_figures" / "output" / "noisy_instances" / "candidates"
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
DEFAULT_CHROM = "chr1"

# Use condition-matched examples for the reviewer-facing transformation figure.
EXAMPLES = (("12hrs", "untr"), ("12hrs", "vacv"))

LEVELS = (100, 95, 60)
PLANE_AXES = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
COLORS = ("#1f77b4", "#d62728")
SHOW_SHAPE_BACKGROUND = True
SHAPE_CLEAN_BLOBS = True
SHAPE_BLOB_MIN_LEN = 15
SHAPE_BLOB_MIN_AREA_FRAC = 0.08

# Publication-facing smoother settings.
CFG_HDR = mpase.CfgHDR(
    n_boot=256,
    sigma_px=1.8,
    density_floor_frac=0.002,
    mass_levels=(1.00, 0.95, 0.60),
)

CFG_PF = mpase.CfgPF(
    frac_levels=(1.00, 0.95, 0.60),
    disk_px=3,
    morph=mpase.CfgMorph(
        closing=2,
        opening=0,
        keep_largest=False,
        fill_holes=True,
    ),
)

def _chrom_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.name)
    return (int(match.group(0)) if match else 10_000, path.name)


def _structure_path(chrom: str, hrs: str, cond: str) -> Path:
    return DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"


def _has_example(chrom: str, examples: Iterable[tuple[str, str]]) -> bool:
    return all(_structure_path(chrom, hrs, cond).exists() for hrs, cond in examples)


def _complete_condition_pairs() -> list[tuple[str, str]]:
    pairs = []
    chrom_dirs = sorted(
        (p for p in DATA_ROOT.iterdir() if p.is_dir()),
        key=_chrom_key,
    )

    for chrom_dir in chrom_dirs:
        for hrs in ("12hrs", "18hrs", "24hrs"):
            if _has_example(chrom_dir.name, ((hrs, "untr"), (hrs, "vacv"))):
                pairs.append((chrom_dir.name, hrs))

    return pairs


def choose_chromosome(preferred: str = DEFAULT_CHROM) -> str:
    if _has_example(preferred, EXAMPLES):
        return preferred

    chrom_dirs = sorted(
        (p for p in DATA_ROOT.iterdir() if p.is_dir()),
        key=_chrom_key,
    )

    for chrom_dir in chrom_dirs:
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


def collect_pair_inputs(chrom: str, hrs: str) -> tuple[list[str], list[str], list[str]]:
    csvs, labels, pretty = [], [], []

    for cond in ("untr", "vacv"):
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


def run_pair(chrom: str, hrs: str, plane: str) -> tuple[dict, list[str], list[np.ndarray], list[str]]:
    csvs, labels, pretty = collect_pair_inputs(chrom, hrs)

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


def _style_scatter_box(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


def _flip_raw_scatter_y(axes) -> None:
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)


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
        _style_scatter_box(ax)

    _set_shared_2d_limits(axes, point_sets2d)
    _flip_raw_scatter_y(axes)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_projection_csvs(
    point_sets2d: list[np.ndarray],
    labels: list[str],
    plane: str,
    out_dir: Path,
    prefix: str,
) -> None:
    for pts, label in zip(point_sets2d, labels):
        out = out_dir / f"{prefix}_{plane}_{_safe_name(label)}.csv"
        pd.DataFrame(pts, columns=[plane[0], plane[1]]).to_csv(out, index=False)


def aligned_points_as_mask_pixels(result: dict, plane: str, labels: list[str]) -> tuple[list[np.ndarray], int, int]:
    projection = result["projections"][plane]
    xs, ys = projection["xs"], projection["ys"]
    point_sets = []

    for label in labels:
        x_idx, y_idx = points_to_pixel_indices(projection["sets"][label], xs, ys)
        point_sets.append(np.column_stack((x_idx, y_idx)))

    return point_sets, len(xs), len(ys)


def save_pixel_point_pair(
    point_sets2d: list[np.ndarray],
    pretty: list[str],
    title: str,
    out_path: Path,
    dpi: int,
    width: int,
    height: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharex=True, sharey=True)

    for ax, pts, label, color in zip(axes, point_sets2d, pretty, COLORS):
        ax.scatter(pts[:, 0], pts[:, 1], s=4.0, alpha=0.65, color=color)
        ax.set_title(label)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        _style_scatter_box(ax)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _shape_for(result: dict, kind: str, plane: str, level: int, label: str):
    return result.get("shapes", {}).get(kind, {}).get(plane, {}).get(level, {}).get(label)


def _background_for(result: dict, plane: str, label: str):
    if "background_by_label" in result and plane in result["background_by_label"]:
        bg_single = result["background_by_label"][plane].get(label)
        if bg_single is not None:
            return bg_single

    return result.get("background", {}).get(plane)


def plot_single_shape(
    ax,
    shape: dict,
    bg_single: np.ndarray | None,
    title: str,
    color: str,
) -> None:
    mpase_plot_single(
        ax,
        shape,
        bg_single if SHOW_SHAPE_BACKGROUND else None,
        title,
        color=color,
        clean=SHAPE_CLEAN_BLOBS,
        blob_min_len=SHAPE_BLOB_MIN_LEN,
        blob_min_area_frac=SHAPE_BLOB_MIN_AREA_FRAC,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlim(0, shape["mask"].shape[1])
    ax.set_ylim(shape["mask"].shape[0], 0)
    ax.set_aspect("equal")


def save_shape_pair(
    result: dict,
    pretty: list[str],
    plane: str,
    kind: str,
    level: int,
    out_dir: Path,
    dpi: int,
) -> None:
    labels = result["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))

    for ax, label, display, color in zip(axes, labels, pretty, COLORS):
        shape = _shape_for(result, kind, plane, level, label)

        if shape is None:
            ax.text(
                0.5,
                0.5,
                f"No {kind} {level}% for {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        plot_single_shape(
            ax,
            shape,
            _background_for(result, plane, label),
            display,
            color,
        )

    fig.suptitle(f"{plane} - {kind} {level}%")
    fig.savefig(out_dir / f"{kind}_{plane}_{level}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _shape_noise_score(shape: dict) -> tuple[float, int, int, int]:
    mask = np.asarray(shape["mask"], dtype=bool)
    _, components = ndi.label(mask)
    contours = all_contours_from_bool(mask, min_len=15, min_area_frac=0.05)
    largest = 0

    if components:
        labeled, count = ndi.label(mask)
        sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
        largest = int(np.max(sizes)) if len(sizes) else 0

    small_pixels = int(mask.sum()) - largest
    score = float(components + len(contours) * 2 + small_pixels / 250.0)
    return score, int(components), int(len(contours)), int(mask.sum())


def score_candidate(result: dict, plane: str, level: int = 100) -> dict:
    rows = []

    for label in result["labels"]:
        hdr_shape = _shape_for(result, "hdr", plane, level, label)
        pf_shape = _shape_for(result, "point_fraction", plane, level, label)

        hdr_score, hdr_components, hdr_contours, hdr_pixels = _shape_noise_score(hdr_shape)
        pf_score, pf_components, pf_contours, pf_pixels = _shape_noise_score(pf_shape)
        score = hdr_score + 0.5 * pf_score

        rows.append(
            {
                "label": label,
                "score": score,
                f"hdr{level}_components": hdr_components,
                f"hdr{level}_contours": hdr_contours,
                f"hdr{level}_pixels": hdr_pixels,
                f"pf{level}_components": pf_components,
                f"pf{level}_contours": pf_contours,
                f"pf{level}_pixels": pf_pixels,
            }
        )

    return max(rows, key=lambda row: row["score"])


def save_candidate_figure(
    result: dict,
    pretty: list[str],
    raw_centered: list[np.ndarray],
    plane: str,
    chrom: str,
    hrs: str,
    out_dir: Path,
    dpi: int,
) -> None:
    labels = result["labels"]
    i, j = PLANE_AXES[plane]
    raw2d = [pts[:, [i, j]] for pts in raw_centered]
    projection = result["projections"][plane]
    xs, ys = projection["xs"], projection["ys"]
    aligned_pixels, _, _ = aligned_points_as_mask_pixels(result, plane, labels)

    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.2))
    colors = dict(zip(labels, COLORS))

    for row, (label, display) in enumerate(zip(labels, pretty)):
        color = colors[label]

        axes[row, 0].scatter(raw2d[row][:, 0], raw2d[row][:, 1], s=4, alpha=0.6, color=color)
        axes[row, 0].set_title(f"{display} raw", fontsize=10, pad=2)
        _style_scatter_box(axes[row, 0])

        axes[row, 1].scatter(aligned_pixels[row][:, 0], aligned_pixels[row][:, 1], s=4, alpha=0.6, color=color)
        axes[row, 1].set_title(f"{display} aligned", fontsize=10, pad=2)
        axes[row, 1].set_xlim(0, len(xs))
        axes[row, 1].set_ylim(len(ys), 0)
        _style_scatter_box(axes[row, 1])

        for col, (kind, level) in enumerate((("hdr", 100), ("point_fraction", 100)), start=2):
            shape = _shape_for(result, kind, plane, level, label)
            plot_single_shape(
                axes[row, col],
                shape,
                _background_for(result, plane, label),
                f"{display} {kind} {level}%",
                color,
            )

    _set_shared_2d_limits(axes[:, 0], raw2d)
    _flip_raw_scatter_y(axes[:, 0])

    fig.suptitle(
        f"Noisy-instance candidate: {chrom} {hrs.replace('hrs', 'h')} ({plane}, 100%)",
        fontsize=12,
        y=0.965,
    )
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.075, top=0.86, wspace=0.08, hspace=0.24)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"candidate_{chrom}_{hrs}_{plane}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def find_noisy_candidates(plane: str, out_dir: Path, dpi: int, top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob(f"candidate_*_{plane}.png"):
        old_png.unlink()
    old_scores = out_dir / f"candidate_scores_{plane}.csv"
    if old_scores.exists():
        old_scores.unlink()

    rows = []

    for chrom, hrs in _complete_condition_pairs():
        result, pretty, raw_centered, _ = run_pair(chrom, hrs, plane)
        row = score_candidate(result, plane)
        row.update({"chrom": chrom, "hrs": hrs})
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary.to_csv(out_dir / f"candidate_scores_{plane}.csv", index=False)

    for _, row in summary.head(top_n).iterrows():
        result, pretty, raw_centered, _ = run_pair(str(row["chrom"]), str(row["hrs"]), plane)
        save_candidate_figure(
            result,
            pretty,
            raw_centered,
            plane,
            str(row["chrom"]),
            str(row["hrs"]),
            out_dir,
            dpi,
        )

    print(f"Saved candidate score table: {out_dir / f'candidate_scores_{plane}.csv'}")
    print(f"Saved top {top_n} candidate figures: {out_dir}")


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


def save_images(
    result: dict,
    pretty: list[str],
    raw_centered: list[np.ndarray],
    plane: str,
    out_dir: Path,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = result["labels"]
    i, j = PLANE_AXES[plane]

    raw2d = [pts[:, [i, j]] for pts in raw_centered]
    save_point_pair(
        raw2d,
        pretty,
        plane,
        f"{plane} projection before alignment",
        out_dir / f"raw_projection_{plane}.png",
        dpi,
    )
    save_projection_csvs(raw2d, labels, plane, out_dir, "raw_projection")

    aligned2d = [result["projections"][plane]["sets"][label] for label in labels]
    aligned_pixels, width, height = aligned_points_as_mask_pixels(result, plane, labels)
    save_pixel_point_pair(
        aligned_pixels,
        pretty,
        f"{plane} projection after MPASE alignment",
        out_dir / f"projection_{plane}.png",
        dpi,
        width,
        height,
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
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR), help="Candidate output directory")
    parser.add_argument("--find-candidates", action="store_true", help="Scan all complete UNTR/VACV time pairs")
    parser.add_argument("--top-n", default=8, type=int, help="Number of candidate figures to save")
    parser.add_argument("--dpi", default=400, type=int, help="PNG export DPI")
    args = parser.parse_args()

    if args.find_candidates:
        find_noisy_candidates(args.plane, Path(args.candidate_dir), args.dpi, args.top_n)
        print(f"Rerun command: python evaluation/supplementary_noisy_instances.py --find-candidates")
        return

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
