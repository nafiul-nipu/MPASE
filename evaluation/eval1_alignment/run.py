"""
Eval 1 — Alignment Validation

Measures RMSE before and after MPASE alignment given a known rigid perturbation.

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
sys.path.insert(0, os.path.join(_ROOT, "src"))  # for mpase package (editable install fallback)
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))  # for data.generate_synthetic

import mpase
from data.generate_synthetic import SHAPES, apply_rigid, random_rotation, random_translation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

N_TRIALS = 20
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SHAPE_NAMES = ["s_shape", "helix", "blob"]


def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def run_synthetic():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    rng = np.random.default_rng(42)

    for shape_name in SHAPE_NAMES:
        gen = SHAPES[shape_name]
        ref = gen(n=500, rng=np.random.default_rng(0))

        for noise in NOISE_LEVELS:
            for trial in range(N_TRIALS):
                R = random_rotation(rng)
                t = random_translation(rng, scale=2.0)
                perturbed = apply_rigid(ref, R, t, noise_frac=noise, rng=rng)

                rmse_before = rmse(ref, perturbed)

                result = mpase.align_points(points_list=[ref, perturbed], labels=["A", "B"])
                aligned_ref, aligned_perturbed = result["aligned_points"]
                rmse_after = rmse(aligned_ref, aligned_perturbed)

                rows.append({
                    "shape": shape_name,
                    "noise_level": noise,
                    "trial": trial,
                    "rmse_before": rmse_before,
                    "rmse_after": rmse_after,
                })
                print(f"  [{shape_name}] noise={noise:.0%} trial={trial:02d}  "
                      f"before={rmse_before:.4f}  after={rmse_after:.4f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "rmse_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df


def run_real(csv_paths: list[str], labels: list[str]):
    """Align two real point cloud CSVs and report RMSE."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = mpase.run(
        csv_list=csv_paths,
        labels=labels,
    )

    pts_A, pts_B = result["aligned_points"]
    r = rmse(pts_A, pts_B)
    print(f"RMSE between aligned real point clouds ({labels[0]} vs {labels[1]}): {r:.4f}")

    rows = [{"shape": "real", "noise_level": 0.0, "trial": 0,
             "rmse_before": float("nan"), "rmse_after": r}]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "rmse_results_real.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return df


def plot_results(df: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_melted = df.melt(
        id_vars=["shape", "noise_level", "trial"],
        value_vars=["rmse_before", "rmse_after"],
        var_name="stage",
        value_name="RMSE",
    )
    df_melted["noise_label"] = df_melted["noise_level"].map(
        lambda x: f"{int(x*100)}%"
    )

    g = sns.FacetGrid(df_melted, col="shape", height=4, sharey=False)
    g.map_dataframe(
        sns.boxplot,
        x="noise_label",
        y="RMSE",
        hue="stage",
        palette={"rmse_before": "#e07070", "rmse_after": "#70a0e0"},
        order=[f"{int(n*100)}%" for n in NOISE_LEVELS],
    )
    g.add_legend(title="Stage")
    g.set_axis_labels("Noise level", "RMSE")
    g.set_titles(col_template="{col_name}")
    plt.suptitle("Alignment RMSE Before vs After (Eval 1)", y=1.02)
    out = os.path.join(FIGURES_DIR, "rmse_by_noise.png")
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
        run_real(args.csv, labels)


if __name__ == "__main__":
    main()
