"""
Eval 3 — Bootstrap Stability

Measures pairwise IoU between HDR masks across N different random seeds.
High mean IoU + low std = stable/robust shapes.

Usage:
    python run.py --mode synthetic
    python run.py --mode real --csv path/to/A.csv path/to/B.csv --labels A B
"""

import argparse
import itertools
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
from data.generate_synthetic import SHAPES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

N_SEEDS = 30
SHAPE_NAMES = ["s_shape", "helix"]
LEVELS = [100, 95, 80, 60]
PLANES = ["XY", "YZ", "XZ"]


def iou_masks(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 1.0


def collect_masks(result: dict, variant: str, label: str) -> dict:
    """Return {(plane, level): bool mask} for a given variant and label."""
    out = {}
    shapes = result["shapes"].get(variant, {})
    for plane, by_level in shapes.items():
        for level, by_label in by_level.items():
            sp = by_label.get(label)
            if sp is not None and sp["mask"] is not None:
                out[(plane, level)] = sp["mask"]
    return out


def run_for_pair(pts_A: np.ndarray, pts_B: np.ndarray,
                 label_a: str, label_b: str, source_name: str) -> pd.DataFrame:
    rows = []
    cfg_common = mpase.CfgCommon()
    cfg_pf = mpase.CfgPF(frac_levels=tuple(l / 100 for l in LEVELS))

    # Collect one mask-dict per seed per label
    masks_by_seed: list[dict] = []
    for seed in range(N_SEEDS):
        cfg_hdr = mpase.CfgHDR(
            n_boot=128,
            mass_levels=tuple(l / 100 for l in LEVELS),
            rng_seed=seed,
        )
        result = mpase.run(
            points_list=[pts_A, pts_B],
            labels=[label_a, label_b],
            cfg_common=cfg_common,
            cfg_hdr=cfg_hdr,
            cfg_pf=cfg_pf,
        )
        masks_by_seed.append({
            "hdr_A":  collect_masks(result, "hdr", label_a),
            "hdr_B":  collect_masks(result, "hdr", label_b),
            "pf_A":   collect_masks(result, "point_fraction", label_a),
            "pf_B":   collect_masks(result, "point_fraction", label_b),
        })
        print(f"  [{source_name}] seed {seed:02d} done")

    # Pairwise IoU across seeds
    for (i, mi), (j, mj) in itertools.combinations(enumerate(masks_by_seed), 2):
        for key in ["hdr_A", "hdr_B", "pf_A", "pf_B"]:
            for (plane, level), mask_i in mi[key].items():
                mask_j = mj[key].get((plane, level))
                if mask_j is None:
                    continue
                rows.append({
                    "source": source_name,
                    "key": key,
                    "plane": plane,
                    "level": level,
                    "seed_i": i,
                    "seed_j": j,
                    "IoU": iou_masks(mask_i, mask_j),
                })

    return pd.DataFrame(rows)


def run_synthetic() -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_dfs = []

    for shape_name in SHAPE_NAMES:
        gen = SHAPES[shape_name]
        ref = gen(n=500, rng=np.random.default_rng(0))
        df = run_for_pair(ref, ref, "A", "B", source_name=shape_name)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    out_csv = os.path.join(RESULTS_DIR, "stability_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df


def run_real(csv_paths: list[str], labels: list[str]) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import pandas as _pd
    pts_list = []
    for path in csv_paths:
        raw = _pd.read_csv(path)[["x", "y", "z"]].dropna().values.astype(np.float32)
        pts_list.append(raw)

    df = run_for_pair(pts_list[0], pts_list[1], labels[0], labels[1], source_name="real")
    out_csv = os.path.join(RESULTS_DIR, "stability_results_real.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return df


def plot_results(df: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # IoU std dev per level
    summary = (
        df.groupby(["source", "key", "plane", "level"])["IoU"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    for source in df["source"].unique():
        sub = summary[summary["source"] == source]

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=sub, x="level", y="std", hue="key", style="plane",
                     markers=True, ax=ax)
        ax.set_title(f"IoU std dev across {N_SEEDS} bootstrap seeds — {source}")
        ax.set_xlabel("Level (%)")
        ax.set_ylabel("IoU std dev")
        out = os.path.join(FIGURES_DIR, f"iou_variance_by_level_{source}.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")

    # Pairwise IoU heatmap for one representative case
    rep_source = SHAPE_NAMES[0] if SHAPE_NAMES[0] in df["source"].values else df["source"].iloc[0]
    rep_key = "hdr_A"
    rep_plane = "XY"
    rep_level = 95
    sub = df[
        (df["source"] == rep_source) &
        (df["key"] == rep_key) &
        (df["plane"] == rep_plane) &
        (df["level"] == rep_level)
    ]
    if not sub.empty:
        mat = np.zeros((N_SEEDS, N_SEEDS))
        np.fill_diagonal(mat, 1.0)
        for _, row in sub.iterrows():
            mat[int(row["seed_i"]), int(row["seed_j"])] = row["IoU"]
            mat[int(row["seed_j"]), int(row["seed_i"])] = row["IoU"]
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(mat, vmin=0, vmax=1, cmap="Blues", ax=ax,
                    xticklabels=5, yticklabels=5)
        ax.set_title(f"Pairwise IoU across seeds ({rep_source}, {rep_key}, {rep_plane}, {rep_level}%)")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Seed")
        out = os.path.join(FIGURES_DIR, f"iou_heatmap_{rep_source}.png")
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
