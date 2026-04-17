"""
Eval 4 — Multi-Chromosome Generalization

Runs MPASE on real 3D genome structure data from multiple chromosomes.
Collects IoU and Hausdorff per chromosome, plane, and level.

Usage:
    python run.py --data_dir path/to/data --chroms chr1 chr2 chr5 chr10

Expected data layout:
    data_dir/
        chr1/
            condition_A.csv
            condition_B.csv
        chr2/
            ...

Each CSV must have columns: x, y, z
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mpase

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

LEVELS = [100, 95, 80]


def run_chromosome(chrom: str, csv_A: str, csv_B: str) -> list:
    cfg_hdr = mpase.CfgHDR(
        n_boot=128,
        mass_levels=tuple(l / 100 for l in LEVELS),
    )
    cfg_pf = mpase.CfgPF(frac_levels=tuple(l / 100 for l in LEVELS))

    result = mpase.run(
        csv_list=[csv_A, csv_B],
        labels=["A", "B"],
        cfg_hdr=cfg_hdr,
        cfg_pf=cfg_pf,
    )

    rows = []
    for _, row in result["metrics"].iterrows():
        rows.append({
            "chrom": chrom,
            "plane": row["plane"],
            "level": row["level"],
            "variant": row["variant"],
            "IoU": row["IoU"],
            "Hausdorff": row["Hausdorff"],
        })
    return rows


def run_real(data_dir: str, chroms: list) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    for chrom in chroms:
        chrom_dir = os.path.join(data_dir, chrom)
        csv_A = os.path.join(chrom_dir, "condition_A.csv")
        csv_B = os.path.join(chrom_dir, "condition_B.csv")

        if not os.path.exists(csv_A) or not os.path.exists(csv_B):
            print(f"  Skipping {chrom}: missing CSV files in {chrom_dir}")
            continue

        print(f"  Running {chrom} ...")
        try:
            rows = run_chromosome(chrom, csv_A, csv_B)
            all_rows.extend(rows)
            print(f"  {chrom}: {len(rows)} metric rows")
        except Exception as e:
            print(f"  {chrom}: ERROR — {e}")

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, "multichrom_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df


def plot_results(df: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    chroms = sorted(df["chrom"].unique())

    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]

        fig, ax = plt.subplots(figsize=(max(6, len(chroms) * 1.5), 4))
        sns.barplot(data=sub, x="chrom", y="IoU", hue="level",
                    order=chroms, ax=ax, palette="Blues_d")
        ax.set_title(f"IoU by chromosome — {variant}")
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("IoU")
        ax.legend(title="Level (%)")
        out = os.path.join(FIGURES_DIR, f"iou_by_chrom_{variant}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")

        pivot = sub.pivot_table(index="chrom", columns="level", values="IoU", aggfunc="mean")
        pivot = pivot.reindex(index=chroms)
        fig, ax = plt.subplots(figsize=(max(5, len(LEVELS) * 1.2), max(4, len(chroms) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Mean IoU heatmap: chromosomes x levels — {variant}")
        ax.set_xlabel("Level (%)")
        ax.set_ylabel("Chromosome")
        out = os.path.join(FIGURES_DIR, f"summary_heatmap_{variant}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing one subdir per chromosome")
    parser.add_argument("--chroms", nargs="+", default=["chr1", "chr2", "chr5", "chr10"],
                        help="Chromosome names (must match subdir names)")
    args = parser.parse_args()

    df = run_real(args.data_dir, args.chroms)
    if not df.empty:
        plot_results(df)
    else:
        print("No results — check CSV files exist under data_dir/chrom_name/.")


if __name__ == "__main__":
    main()
