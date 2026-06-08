"""
Generate a supplementary MPASE figure for reviewer-facing noisy instances.

The figure uses one real chromosome example and shows:
  A. Centered point clouds before PCA/ICP alignment.
  B. Point clouds after MPASE PCA/ICP alignment.
  C. HDR shape abstraction at 60% and 95%.
  D. PF shape abstraction at 60% and 95%, when available.

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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mpase
from mpase.metrics_calculation import all_contours_from_bool


DATA_ROOT = ROOT / "evaluation" / "data" / "all_structure_files"
OUT_DIR = ROOT / "evaluation" / "supplementary_figures" / "output"
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
DEFAULT_CHROM = "chr1"
EXAMPLES = (("12hrs", "untr"), ("18hrs", "vacv"))
PLANE_AXES = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
COLORS = {
    "untr": "#2b6cb0",
    "vacv": "#c2410c",
}


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


def run_example(chrom: str, plane: str) -> dict:
    csvs, labels, _ = collect_inputs(chrom)
    return mpase.run(
        csv_list=csvs,
        labels=labels,
        xyz_cols=XYZ_COLS,
        id_col="gene_name",
        cfg_common=mpase.CfgCommon(),
        cfg_hdr=mpase.CfgHDR(n_boot=256, mass_levels=(0.95, 0.60)),
        cfg_pf=mpase.CfgPF(frac_levels=(0.95, 0.60)),
        planes=(plane,),
    )


def export_point_tables(result: dict, out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    ids_by_label = result.get("ids_by_label", {})

    for key, arrays in (
        ("raw_centered", result.get("raw_centered_points", [])),
        ("aligned", result.get("aligned_points", [])),
    ):
        for label, points in zip(labels, arrays):
            df = pd.DataFrame(np.asarray(points), columns=["x", "y", "z"])
            ids = ids_by_label.get(label)
            if ids is not None and len(ids) == len(df):
                df.insert(0, "gene_id", ids)
            df.to_csv(data_dir / f"{_safe_name(label)}_{key}.csv", index=False)


def _set_point_limits(ax, point_sets: list[np.ndarray]) -> None:
    stacked = np.vstack(point_sets)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1e-8)
    pad = radius * 0.08
    ax.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax.set_ylim(center[1] - radius - pad, center[1] + radius + pad)


def plot_points(ax, point_sets: list[np.ndarray], labels: list[str], plane: str, title: str) -> None:
    i, j = PLANE_AXES[plane]
    projected = [np.asarray(points)[:, [i, j]] for points in point_sets]
    for points2d, label in zip(projected, labels):
        cond = "vacv" if "VACV" in label else "untr"
        ax.scatter(
            points2d[:, 0],
            points2d[:, 1],
            s=8,
            alpha=0.62,
            linewidths=0,
            color=COLORS[cond],
            label=label,
        )
    _set_point_limits(ax, projected)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _shape_for(result: dict, kind: str, plane: str, level: int, label: str):
    return result.get("shapes", {}).get(kind, {}).get(plane, {}).get(level, {}).get(label)


def plot_shape_panel(ax, result: dict, labels: list[str], pretty: list[str], plane: str, kind: str, title: str) -> bool:
    levels = (95, 60)
    linestyles = {95: "-", 60: "--"}
    any_shape = False

    for label, display in zip(labels, pretty):
        cond = "vacv" if "VACV" in display else "untr"
        for level in levels:
            shape = _shape_for(result, kind, plane, level, label)
            if shape is None:
                continue
            any_shape = True
            contours = all_contours_from_bool(shape["mask"], min_len=10, min_area_frac=0.0)
            for contour in contours:
                ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    color=COLORS[cond],
                    linestyle=linestyles[level],
                    linewidth=2.0 if level == 95 else 1.7,
                    alpha=0.95 if level == 95 else 0.75,
                )

    ax.set_title(title, fontsize=11, fontweight="bold")
    bg = result["background"].get(plane)
    if bg is not None:
        ny, nx = bg.shape
        ax.set_xlim(0, nx)
        ax.set_ylim(ny, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    return any_shape


def make_figure(result: dict, pretty: list[str], plane: str, out_dir: Path, dpi: int) -> tuple[Path, Path]:
    labels = result["labels"]
    panel_specs = [
        ("A", "raw", "Raw centered point clouds"),
        ("B", "aligned", "Aligned point clouds"),
        ("C", "hdr", "HDR abstraction"),
    ]
    if result.get("shapes", {}).get("point_fraction", {}).get(plane):
        panel_specs.append(("D", "point_fraction", "PF abstraction"))

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(4.3 * len(panel_specs), 4.2), constrained_layout=True)
    if len(panel_specs) == 1:
        axes = [axes]

    for ax, (letter, kind, title) in zip(axes, panel_specs):
        if kind == "raw":
            plot_points(ax, result["raw_centered_points"], pretty, plane, f"{letter}. {title}")
        elif kind == "aligned":
            plot_points(ax, result["aligned_points"], pretty, plane, f"{letter}. {title}")
        else:
            plot_shape_panel(ax, result, labels, pretty, plane, kind, f"{letter}. {title} ({plane})")

    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{labels[0].split('_')[0].upper()} example: noisy instances, MPASE alignment, and derived shapes ({plane})",
        fontsize=12,
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["untr"], markersize=6, label=pretty[0]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["vacv"], markersize=6, label=pretty[1]),
        Line2D([0], [0], color="0.25", lw=2, linestyle="-", label="95% shape"),
        Line2D([0], [0], color="0.25", lw=2, linestyle="--", label="60% shape"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"supp_noisy_instances_{labels[0].split('_')[0]}_{plane}.png"
    pdf = out_dir / f"supp_noisy_instances_{labels[0].split('_')[0]}_{plane}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", default=DEFAULT_CHROM, help="Preferred chromosome, default: chr1")
    parser.add_argument("--plane", default="YZ", choices=sorted(PLANE_AXES), help="Projection plane")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for the figure")
    parser.add_argument("--dpi", default=400, type=int, help="PNG export DPI")
    args = parser.parse_args()

    chrom = choose_chromosome(args.chrom)
    _, labels, pretty = collect_inputs(chrom)
    result = run_example(chrom, args.plane)

    out_dir = Path(args.out_dir)
    export_point_tables(result, out_dir)
    png, pdf = make_figure(result, pretty, args.plane, out_dir, args.dpi)

    print(f"Chromosome: {chrom}")
    print("Examples: " + ", ".join(labels))
    print(f"Saved PNG: {png}")
    print(f"Saved PDF: {pdf}")
    print(f"Saved point tables: {out_dir / 'data'}")


if __name__ == "__main__":
    main()
