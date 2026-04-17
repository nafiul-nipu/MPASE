"""
Summary figures for paper.

Reads CSVs produced by eval1–eval3 and generates combined paper-ready plots.

Usage:
    python summary_figures.py
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EVAL_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(EVAL_DIR, "figures")


def load_csv(rel_path: str) -> pd.DataFrame:
    path = os.path.join(EVAL_DIR, rel_path)
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def figure_eval1_summary():
    df = load_csv("eval1_alignment/results/rmse_results.csv")
    if df.empty:
        return

    df["noise_label"] = df["noise_level"].map(lambda x: f"{int(x*100)}%")
    df_m = df.melt(
        id_vars=["shape", "noise_label"],
        value_vars=["rmse_before", "rmse_after"],
        var_name="stage", value_name="RMSE",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_m, x="noise_label", y="RMSE", hue="stage",
                palette={"rmse_before": "#e07070", "rmse_after": "#70a0e0"}, ax=ax)
    ax.set_title("Eval 1 — Alignment RMSE Before vs After")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("RMSE")
    out = os.path.join(FIGURES_DIR, "paper_eval1_rmse.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out}")


def figure_eval2_summary():
    df = load_csv("eval2_ablation/results/ablation_results.csv")
    if df.empty:
        return

    for variant in df["variant"].unique():
        sub = df[(df["variant"] == variant) & (df["plane"] == "XY")]
        order = ["no_align", "pca_only", "pca_icp"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(data=sub, x="align_mode", y="IoU", order=order,
                    palette="Set2", ax=axes[0])
        axes[0].set_title("IoU by alignment method")
        axes[0].set_xlabel("Alignment mode")
        sns.boxplot(data=sub, x="align_mode", y="Hausdorff", order=order,
                    palette="Set2", ax=axes[1])
        axes[1].set_title("Hausdorff by alignment method")
        axes[1].set_xlabel("Alignment mode")
        fig.suptitle(f"Eval 2 — Ablation Study ({variant}, XY plane)")
        out = os.path.join(FIGURES_DIR, f"paper_eval2_ablation_{variant}.png")
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved {out}")


def figure_eval3_summary():
    df = load_csv("eval3_stability/results/stability_results.csv")
    if df.empty:
        return

    summary = (
        df.groupby(["source", "key", "level"])["IoU"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    for source in summary["source"].unique():
        sub = summary[summary["source"] == source]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=sub, x="level", y="std", hue="key",
                     markers=True, ax=ax)
        ax.set_title(f"Eval 3 — IoU stability across seeds ({source})")
        ax.set_xlabel("Level (%)")
        ax.set_ylabel("IoU std dev")
        out = os.path.join(FIGURES_DIR, f"paper_eval3_stability_{source}.png")
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure_eval1_summary()
    figure_eval2_summary()
    figure_eval3_summary()
    print("Done.")


if __name__ == "__main__":
    main()
