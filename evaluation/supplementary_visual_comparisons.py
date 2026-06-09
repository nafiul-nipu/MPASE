"""
Generate supplementary side-by-side visual comparison figures from MPASE.

This figure complements quantitative IoU/meanNN usage-case tables with
representative chromosome-level HDR and PF silhouettes.

Usage:
    python evaluation/supplementary_visual_comparisons.py
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
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mpase
from mpase.metrics_calculation import contour_from_bool


DATA_ROOT = ROOT / "evaluation" / "data" / "all_structure_files"
METRICS_PATH = ROOT / "evaluation" / "eval_shape_representation" / "results" / "all_chroms_all_pairs_iou.csv"
OUT_DIR = ROOT / "evaluation" / "supplementary_figures" / "output"
PANEL_DIR = OUT_DIR / "visual_comparison_timecourses"

TIMES = ("12hrs", "18hrs", "24hrs")
CONDS = ("untr", "vacv")
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
PLANE = "YZ"
LEVEL = 100
EXCLUDE_FROM_AUTO_SELECTION = {"chr29"}
PREFERRED_CHROMS = ("chr1", "chr14", "chr20")

CFG_COMMON = mpase.CfgCommon(icp_iters=30)
CFG_HDR = mpase.CfgHDR(
    n_boot=256,
    sigma_px=1.8,
    density_floor_frac=0.002,
    mass_levels=(1.00,),
)
CFG_PF = mpase.CfgPF(
    frac_levels=(1.00,),
    morph=mpase.CfgMorph(
        closing=2,
        opening=2,
        keep_largest=True,
        fill_holes=True,
    ),
)

CAPTION = (
    "Supplementary Fig. X. Side-by-side visual comparisons of representative "
    "chromosome shapes corresponding to the quantitative usage-case results. "
    "Each block shows chromosome-level HDR shapes in the YZ projection at the "
    "100% abstraction level for UNTR and VACV conditions across 12h, 18h, and "
    "24h. The selected chromosomes illustrate a range of structural behaviors, "
    "including relatively stable patterns and stronger infection-associated "
    "changes. These visual comparisons complement the IoU and meanNN values "
    "reported in the supplementary table by showing the shape differences "
    "underlying the quantitative trends."
)


def _chrom_key(chrom: str) -> tuple[int, str]:
    match = re.search(r"\d+", chrom)
    return (int(match.group(0)) if match else 10_000, chrom)


def _structure_path(chrom: str, hrs: str, cond: str) -> Path:
    return DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"


def _complete_chromosomes() -> list[str]:
    chroms = []
    for chrom_dir in DATA_ROOT.iterdir():
        if not chrom_dir.is_dir():
            continue
        chrom = chrom_dir.name
        if chrom in EXCLUDE_FROM_AUTO_SELECTION:
            continue
        if all(_structure_path(chrom, hrs, cond).exists() for hrs in TIMES for cond in CONDS):
            chroms.append(chrom)
    return sorted(chroms, key=_chrom_key)


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def select_chromosomes(n_chroms: int = 3) -> tuple[list[str], str]:
    complete = _complete_chromosomes()
    if not complete:
        raise FileNotFoundError(f"No complete chromosomes found in {DATA_ROOT}")

    preferred = [chrom for chrom in PREFERRED_CHROMS if chrom in complete]
    if len(preferred) >= n_chroms:
        return preferred[:n_chroms], "manual: preferred chromosomes"

    selected = ["chr1"] if "chr1" in complete else [complete[0]]

    if METRICS_PATH.exists():
        df = pd.read_csv(METRICS_PATH)
        required = {"chrom", "category", "representation", "level", "IoU"}
        if required.issubset(df.columns):
            sub = df[
                (df["chrom"].isin(complete))
                & (df["representation"].astype(str).str.upper() == "HDR")
                & (df["level"] == LEVEL)
            ].copy()

            stable = sub[sub["category"] == "Stability"].sort_values("IoU", ascending=False)
            strong = sub[sub["category"] == "Condition difference"].sort_values("IoU", ascending=True)

            if not stable.empty:
                selected.append(str(stable.iloc[0]["chrom"]))
            if not strong.empty:
                selected.append(str(strong.iloc[0]["chrom"]))

            selected = _unique_keep_order(c for c in selected if c in complete)
            if len(selected) >= n_chroms:
                return selected[:n_chroms], f"metrics: {METRICS_PATH}"

    fallback = _unique_keep_order(selected + [c for c in complete if c not in selected])
    return fallback[:n_chroms], "fallback: automatic metric-based selection was not possible"


def collect_inputs(chrom: str) -> tuple[list[str], list[str]]:
    csvs, labels = [], []
    for cond in CONDS:
        for hrs in TIMES:
            csvs.append(str(_structure_path(chrom, hrs, cond)))
            labels.append(f"{chrom}_{hrs}_{cond}")
    return csvs, labels


def run_chromosome(chrom: str) -> dict:
    csvs, labels = collect_inputs(chrom)
    return mpase.run(
        csv_list=csvs,
        labels=labels,
        xyz_cols=XYZ_COLS,
        id_col="gene_name",
        cfg_common=CFG_COMMON,
        cfg_hdr=CFG_HDR,
        cfg_pf=CFG_PF,
        planes=(PLANE,),
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count <= 1:
        return mask.astype(bool)

    sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    return labels == largest


def _background_for(result: dict, label: str) -> np.ndarray | None:
    if "background_by_label" in result and PLANE in result["background_by_label"]:
        bg_single = result["background_by_label"][PLANE].get(label)
        if bg_single is not None:
            return bg_single
    return result.get("background", {}).get(PLANE)


def _plot_shape(ax, shape: dict, bg_single: np.ndarray | None, color: str, title: str) -> None:
    if bg_single is not None:
        ax.imshow(bg_single, cmap="gray", alpha=0.18, interpolation="nearest")

    mask = _largest_component(shape["mask"].astype(bool))
    contour = contour_from_bool(mask)
    rgb = to_rgb(color)
    rgba = (*rgb, 0.18)

    ax.imshow(
        np.dstack(
            [
                np.full(mask.shape, rgb[0]),
                np.full(mask.shape, rgb[1]),
                np.full(mask.shape, rgb[2]),
                mask.astype(float) * rgba[3],
            ]
        ),
        interpolation="nearest",
    )

    if contour is not None:
        ax.plot(contour[:, 1], contour[:, 0], color=color, lw=1.8)

    ax.set_title(title, fontsize=8, pad=1)
    ax.set_xlim(0, shape["mask"].shape[1])
    ax.set_ylim(shape["mask"].shape[0], 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _kind_label(kind: str) -> str:
    return "PF" if kind == "point_fraction" else kind.upper()


def save_timecourse_panels(
    results: dict[str, dict],
    chroms: list[str],
    panel_dir: Path,
    dpi: int,
) -> list[Path]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    colors = {"untr": "#1f77b4", "vacv": "#d62728"}
    saved: list[Path] = []

    for chrom in chroms:
        result = results[chrom]
        for kind in ("hdr", "point_fraction"):
            for cond in CONDS:
                fig, axes = plt.subplots(1, 3, figsize=(3.95, 1.85), constrained_layout=False)
                for ax, hrs in zip(axes, TIMES):
                    label = f"{chrom}_{hrs}_{cond}"
                    shape = result["shapes"][kind][PLANE][LEVEL][label]
                    title = f"{cond.upper()} {hrs.replace('hrs', 'h')}"
                    _plot_shape(ax, shape, _background_for(result, label), colors[cond], title)

                pretty_kind = _kind_label(kind)
                fig.suptitle(f"{chrom} - {pretty_kind} {LEVEL}% - {cond.upper()} - {PLANE}", fontsize=10, y=0.93)
                fig.patch.set_facecolor("white")
                fig.subplots_adjust(left=0.002, right=0.998, bottom=0.035, top=0.78, wspace=-0.18)

                stem = f"{chrom}_{pretty_kind}{LEVEL}_{PLANE}_{cond}_timecourse"
                png_path = panel_dir / f"{stem}.png"
                pdf_path = panel_dir / f"{stem}.pdf"
                fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                saved.extend([png_path, pdf_path])

    return saved


def save_figure(chroms: list[str], out_dir: Path, dpi: int) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {chrom: run_chromosome(chrom) for chrom in chroms}
    n_blocks = len(chroms)
    fig, axes = plt.subplots(
        n_blocks * 2,
        3,
        figsize=(8.4, 4.5 * n_blocks),
        constrained_layout=False,
    )
    if n_blocks == 1:
        axes = np.asarray(axes).reshape(2, 3)

    colors = {"untr": "#1f77b4", "vacv": "#d62728"}

    for block_idx, chrom in enumerate(chroms):
        result = results[chrom]
        block_axes = axes[block_idx * 2 : block_idx * 2 + 2, :]
        block_axes[0, 0].text(
            -0.02,
            1.34,
            chrom,
            transform=block_axes[0, 0].transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
        for row, cond in enumerate(CONDS):
            for col, hrs in enumerate(TIMES):
                ax = block_axes[row, col]
                label = f"{chrom}_{hrs}_{cond}"
                shape = result["shapes"]["hdr"][PLANE][LEVEL][label]
                title = f"{cond.upper()} {hrs.replace('hrs', 'h')}"
                _plot_shape(ax, shape, _background_for(result, label), colors[cond], title)

    fig.suptitle("HDR 100% Shape Comparisons in YZ Projection", fontsize=15, y=0.995)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=3.0, w_pad=0.8)

    png_path = out_dir / "supplementary_visual_comparisons_HDR100_YZ.png"
    pdf_path = out_dir / "supplementary_visual_comparisons_HDR100_YZ.pdf"
    caption_path = out_dir / "supplementary_visual_comparisons_caption.txt"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption_path.write_text(CAPTION + "\n")

    save_timecourse_panels(results, chroms, out_dir / "visual_comparison_timecourses", dpi)

    return png_path, pdf_path, caption_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--n-chroms", default=3, type=int, help="Number of chromosome blocks")
    parser.add_argument("--dpi", default=400, type=int, help="PNG export DPI")
    args = parser.parse_args()

    chroms, selection_mode = select_chromosomes(args.n_chroms)
    png_path, pdf_path, caption_path = save_figure(chroms, Path(args.out_dir), args.dpi)

    print("Selected chromosomes: " + ", ".join(chroms))
    print(f"Selection mode: {selection_mode}")
    if selection_mode.startswith("fallback"):
        print("Warning: automatic metric-based selection was not possible.")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved caption: {caption_path}")
    print(f"Saved split panels folder: {Path(args.out_dir) / 'visual_comparison_timecourses'}")
    print("Rerun command: python evaluation/supplementary_visual_comparisons.py")


if __name__ == "__main__":
    main()
