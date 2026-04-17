"""
Eval 2 — Ablation Study

Compares IoU and Hausdorff for three alignment strategies:
  - no_align  : skip alignment entirely
  - pca_only  : PCA pre-alignment, zero ICP iterations
  - pca_icp   : full pipeline (PCA + ICP)

Usage:
    python run.py --mode synthetic
    python run.py --mode real --csv path/to/A.csv path/to/B.csv --labels A B
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
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))

import mpase
from data.generate_synthetic import SHAPES, apply_rigid, random_rotation, random_translation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

N_TRIALS = 20
SHAPE_NAMES = ["s_shape", "helix", "blob"]
LEVELS = [100, 95, 80, 60]
PLANES = ["XY", "YZ", "XZ"]

ALIGN_MODES = {
    "no_align": mpase.CfgCommon(icp_iters=0, trim_q=0.0),   # skip ICP; centering done in pipeline
    "pca_only": mpase.CfgCommon(icp_iters=0),                # PCA pre-align, no ICP iterations
    "pca_icp":  mpase.CfgCommon(icp_iters=30),               # full pipeline
}


def extract_metrics(result: dict, align_mode: str, shape: str, trial: int) -> list[dict]:
    rows = []
    df_m = result["metrics"]
    for _, row in df_m.iterrows():
        rows.append({
            "shape": shape,
            "align_mode": align_mode,
            "plane": row["plane"],
            "variant": row["variant"],
            "level": row["level"],
            "trial": trial,
            "IoU": row["IoU"],
            "Hausdorff": row["Hausdorff"],
        })
    return rows


def run_synthetic():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    rng = np.random.default_rng(42)

    cfg_hdr = mpase.CfgHDR(n_boot=64, mass_levels=(1.00, 0.95, 0.80, 0.60))
    cfg_pf  = mpase.CfgPF(frac_levels=(1.00, 0.95, 0.80, 0.60))

    for shape_name in SHAPE_NAMES:
        gen = SHAPES[shape_name]
        ref = gen(n=500, rng=np.random.default_rng(0))

        for trial in range(N_TRIALS):
            R = random_rotation(rng)
            t = random_translation(rng, scale=2.0)
            perturbed = apply_rigid(ref, R, t, noise_frac=0.05, rng=rng)

            for mode_name, cfg_common in ALIGN_MODES.items():
                try:
                    result = mpase.run(
                        points_list=[ref, perturbed],
                        labels=["A", "B"],
                        cfg_common=cfg_common,
                        cfg_hdr=cfg_hdr,
                        cfg_pf=cfg_pf,
                    )
                    rows.extend(extract_metrics(result, mode_name, shape_name, trial))
                    print(f"  [{shape_name}] mode={mode_name} trial={trial:02d}  done")
                except Exception as e:
                    print(f"  [{shape_name}] mode={mode_name} trial={trial:02d}  ERROR: {e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df


def run_real(csv_paths: list[str], labels: list[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    cfg_hdr = mpase.CfgHDR(n_boot=64, mass_levels=(1.00, 0.95, 0.80, 0.60))
    cfg_pf  = mpase.CfgPF(frac_levels=(1.00, 0.95, 0.80, 0.60))

    for mode_name, cfg_common in ALIGN_MODES.items():
        try:
            result = mpase.run(
                csv_list=csv_paths,
                labels=labels,
                cfg_common=cfg_common,
                cfg_hdr=cfg_hdr,
                cfg_pf=cfg_pf,
            )
            rows.extend(extract_metrics(result, mode_name, "real", trial=0))
            print(f"  mode={mode_name}  done")
        except Exception as e:
            print(f"  mode={mode_name}  ERROR: {e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "ablation_results_real.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return df


def plot_results(df: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    order = ["no_align", "pca_only", "pca_icp"]

    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]

        fig, axes = plt.subplots(1, len(SHAPE_NAMES), figsize=(5 * len(SHAPE_NAMES), 4), sharey=False)
        for ax, shape in zip(axes, SHAPE_NAMES):
            d = sub[sub["shape"] == shape]
            sns.boxplot(data=d, x="align_mode", y="IoU", order=order, ax=ax,
                        palette="Set2")
            ax.set_title(shape)
            ax.set_xlabel("Alignment mode")
            ax.set_ylabel("IoU")
        fig.suptitle(f"IoU by alignment method — {variant}", y=1.02)
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"iou_by_method_{variant}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")

        fig, axes = plt.subplots(1, len(SHAPE_NAMES), figsize=(5 * len(SHAPE_NAMES), 4), sharey=False)
        for ax, shape in zip(axes, SHAPE_NAMES):
            d = sub[sub["shape"] == shape]
            sns.boxplot(data=d, x="align_mode", y="Hausdorff", order=order, ax=ax,
                        palette="Set2")
            ax.set_title(shape)
            ax.set_xlabel("Alignment mode")
            ax.set_ylabel("Hausdorff distance")
        fig.suptitle(f"Hausdorff by alignment method — {variant}", y=1.02)
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"hausdorff_by_method_{variant}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--csv", nargs="+", help="CSV paths for real mode")
    parser.add_argument("--labels", nargs="+", help="Labels for real mode")
    args = parser.parse_args()

    if args.mode == "synthetic":
        df = run_synthetic()
        plot_results(df)
    else:
        if not args.csv or len(args.csv) < 2:
            parser.error("--mode real requires at least 2 --csv paths")
        labels = args.labels or [f"S{i}" for i in range(len(args.csv))]
        df = run_real(args.csv, labels)
        plot_results(df)


if __name__ == "__main__":
    main()
