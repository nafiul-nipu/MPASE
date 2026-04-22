"""
Shape Representation Analysis — HDR vs Point-Fraction

Three representative pairs for chr1:
  - UNTR 12h vs UNTR 18h  : stability (same condition, adjacent timepoints)
  - VACV 12h vs VACV 18h  : temporal change (infected condition)
  - UNTR vs VACV at 18h   : condition difference (same timepoint)

Settings: PCA+ICP alignment, YZ plane, levels 60/80/95/100%.
Compares IoU and meanNN between HDR and point-fraction representations,
and plots IoU as a function of density level to show multi-scale behavior.

Usage:
    python run.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import mpase

DATA_ROOT   = os.path.join(_ROOT, "evaluation", "data", "all_structure_files")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

CHROM    = "chr1"
PLANE    = "YZ"
LEVELS   = [60, 80, 95, 100]
XYZ_COLS = ("middle_x", "middle_y", "middle_z")

CHROMS = [
    "chr1",
    "chr2","chr3","chr4","chr5","chr6","chr7",
    "chr10","chr12","chr13","chr14","chr15","chr16",
    "chr18","chr19","chr20","chr21","chr22","chr23",
    "chr25","chr26","chr27","chr28","chr29",
]
TIMES = ["12hrs", "18hrs", "24hrs"]
CONDS = ["untr", "vacv"]

CFG_COMMON = mpase.CfgCommon(icp_iters=30)
CFG_HDR    = mpase.CfgHDR(n_boot=256, mass_levels=(0.60, 0.80, 0.95, 1.00))
CFG_PF     = mpase.CfgPF(frac_levels=(0.60, 0.80, 0.95, 1.00))


def _gene_info(chrom, hrs, cond):
    return os.path.join(DATA_ROOT, chrom, hrs, cond,
                        f"structure_{hrs}_{cond}_gene_info.csv")


PAIRS = [
    {
        "name":    "UNTR 12h vs 18h",
        "label":   "Stability",
        "csv_a":   _gene_info(CHROM, "12hrs", "untr"),
        "csv_b":   _gene_info(CHROM, "18hrs", "untr"),
        "label_a": "untr_12h",
        "label_b": "untr_18h",
    },
    {
        "name":    "VACV 12h vs 18h",
        "label":   "Temporal change",
        "csv_a":   _gene_info(CHROM, "12hrs", "vacv"),
        "csv_b":   _gene_info(CHROM, "18hrs", "vacv"),
        "label_a": "vacv_12h",
        "label_b": "vacv_18h",
    },
    {
        "name":    "UNTR vs VACV at 18h",
        "label":   "Condition difference",
        "csv_a":   _gene_info(CHROM, "18hrs", "untr"),
        "csv_b":   _gene_info(CHROM, "18hrs", "vacv"),
        "label_a": "untr_18h",
        "label_b": "vacv_18h",
    },
]


def extract_metric(result: dict, plane: str, level: int, variant: str) -> tuple:
    df = result["metrics"]
    row = df[(df["plane"] == plane) & (df["level"] == level) & (df["variant"] == variant)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["IoU"]), float(row.iloc[0]["meanNN"])


def run_all() -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        print(f"\n  {pair['name']} ({pair['label']})")
        result = mpase.run(
            csv_list=[pair["csv_a"], pair["csv_b"]],
            labels=[pair["label_a"], pair["label_b"]],
            xyz_cols=XYZ_COLS,
            cfg_common=CFG_COMMON,
            cfg_hdr=CFG_HDR,
            cfg_pf=CFG_PF,
        )
        for variant, rep_label in [("hdr", "HDR"), ("point_fraction", "PF")]:
            for level in LEVELS:
                iou, meannn = extract_metric(result, PLANE, level, variant)
                rows.append({
                    "pair":           pair["name"],
                    "category":       pair["label"],
                    "representation": rep_label,
                    "plane":          PLANE,
                    "level":          level,
                    "IoU":            round(iou, 4),
                    "meanNN":         round(meannn, 3),
                })
            iou60, _ = extract_metric(result, PLANE, 60, variant)
            print(f"    {rep_label:4}  IoU@60%={iou60:.3f}")
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def plot_grouped_bars(df: pd.DataFrame):
    sub        = df[df["level"] == 60]
    pair_order = [p["name"] for p in PAIRS]
    palette    = {"HDR": "#4c72b0", "PF": "#dd8452"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        x     = np.arange(len(pair_order))
        width = 0.35
        for i, rep in enumerate(["HDR", "PF"]):
            vals = [sub[(sub["pair"] == p) & (sub["representation"] == rep)][metric].values
                    for p in pair_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=rep, color=palette[rep], alpha=0.88)
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi + (i - 0.5) * width, v + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_order, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — HDR vs PF (YZ, 60%)")
        ax.legend()
        if metric == "IoU":
            ax.set_ylim(0, 0.7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Shape representation comparison — Chr1", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "grouped_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_heatmaps(df: pd.DataFrame):
    sub        = df[df["level"] == 60]
    pair_order = [p["name"] for p in PAIRS]

    for metric, cmap, vmin, vmax in [
        ("IoU",    "RdYlGn",   0, 1),
        ("meanNN", "RdYlGn_r", 0, 20),
    ]:
        pivot = sub.pivot_table(index="pair", columns="representation",
                                values=metric, aggfunc="first")
        pivot = pivot.reindex(pair_order)[["HDR", "PF"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=ax,
                    linewidths=0.8, cbar_kws={"label": metric})
        title_note = "lower = better" if metric == "meanNN" else "higher = better"
        ax.set_title(f"{metric} — HDR vs PF\n(YZ, 60%, Chr1)  [{title_note}]")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"heatmap_{metric.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_delta(df: pd.DataFrame):
    sub        = df[df["level"] == 60]
    pair_order = [p["name"] for p in PAIRS]
    categories = [p["label"] for p in PAIRS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["IoU", "meanNN"]):
        deltas = []
        for p in pair_order:
            hdr = sub[(sub["pair"] == p) & (sub["representation"] == "HDR")][metric].values
            pf  = sub[(sub["pair"] == p) & (sub["representation"] == "PF")][metric].values
            deltas.append(float(pf[0] - hdr[0]) if len(hdr) and len(pf) else np.nan)

        bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax.bar(range(len(pair_order)), deltas, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(pair_order, categories)],
                           rotation=10, ha="right", fontsize=8)
        ax.set_ylabel(f"Δ {metric}  (PF − HDR)")
        direction = "↑ PF better" if metric == "IoU" else "↓ PF better"
        ax.set_title(f"Δ {metric} (PF − HDR)\n{direction}")
        for xi, v in enumerate(deltas):
            if not np.isnan(v):
                ax.text(xi, v + (0.001 if v >= 0 else -0.003),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("PF vs HDR difference — Chr1 (YZ, 60%)", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "delta_pf_vs_hdr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def run_chrom_level() -> pd.DataFrame:
    """
    For each chromosome, pass all 6 CSVs (consistent normalization) and
    extract IoU for UNTR vs VACV at 18h across all levels and both representations.
    """
    rows = []
    for chrom in CHROMS:
        csvs, labels = [], []
        for hrs in TIMES:
            for cond in CONDS:
                p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                                 f"structure_{hrs}_{cond}_gene_info.csv")
                if os.path.exists(p):
                    csvs.append(p)
                    labels.append(f"{chrom}_{hrs}_{cond}")
        if len(csvs) < 2:
            print(f"  {chrom}: missing files, skipping")
            continue

        A_label = f"{chrom}_18hrs_untr"
        B_label = f"{chrom}_18hrs_vacv"

        try:
            result = mpase.run(
                csv_list=csvs,
                labels=labels,
                xyz_cols=XYZ_COLS,
                cfg_common=CFG_COMMON,
                cfg_hdr=CFG_HDR,
                cfg_pf=CFG_PF,
            )
            for variant, rep_label in [("hdr", "HDR"), ("point_fraction", "PF")]:
                for level in LEVELS:
                    df_m = result["metrics"]
                    row  = df_m[
                        (df_m["plane"] == PLANE) & (df_m["level"] == level) &
                        (df_m["variant"] == variant) &
                        (df_m["A"] == A_label) & (df_m["B"] == B_label)
                    ]
                    iou = float(row.iloc[0]["IoU"]) if not row.empty else float("nan")
                    rows.append({
                        "chrom":          chrom,
                        "representation": rep_label,
                        "level":          level,
                        "IoU":            round(iou, 4),
                    })
            print(f"  {chrom}: done")
        except Exception as e:
            print(f"  {chrom}: ERROR — {e}")

    return pd.DataFrame(rows)


def plot_chrom_level_heatmap(df: pd.DataFrame, representation: str):
    """
    Table-style heatmap: chromosomes × HDR/PF levels, cells = IoU.
    Fixed comparison: UNTR vs VACV at 18h, YZ plane.
    """
    HEATMAP_LEVELS = [60, 80, 95]
    sub   = df[(df["representation"] == representation) & (df["level"].isin(HEATMAP_LEVELS))]
    pivot = sub.pivot_table(index="chrom", columns="level", values="IoU")
    pivot = pivot[[60, 80, 95]]
    pivot.columns = ["60%", "80%", "95%"]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c: int(c.replace("chr", ""))))

    fig, ax = plt.subplots(figsize=(5, max(5, len(pivot) * 0.38)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, cbar_kws={"label": "IoU"})
    ax.set_title(f"IoU — UNTR vs VACV at 18h ({representation})\n"
                 f"YZ plane | PCA+ICP alignment\n"
                 f"lower = greater structural difference")
    ax.set_xlabel("HDR level" if representation == "HDR" else "PF level")
    ax.set_ylabel("Chromosome")
    plt.tight_layout()
    fname = f"chrom_level_heatmap_{representation.lower()}.png"
    out   = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_iou_vs_level(df: pd.DataFrame, representation: str):
    """
    Line plot: IoU vs density level for each comparison pair.
    Shows how similarity varies from dense core (60%) to near-global (100%).
    """
    sub        = df[df["representation"] == representation]
    pair_order = [p["name"] for p in PAIRS]
    categories = [p["label"] for p in PAIRS]
    palette    = sns.color_palette("tab10", len(pair_order))
    x_labels   = [f"{l}%" for l in LEVELS]

    fig, ax = plt.subplots(figsize=(7, 4))
    for color, pair_name, cat in zip(palette, pair_order, categories):
        psub = sub[sub["pair"] == pair_name].sort_values("level")
        iou_vals = [psub[psub["level"] == l]["IoU"].values for l in LEVELS]
        iou_vals = [v[0] if len(v) else np.nan for v in iou_vals]
        ax.plot(range(len(LEVELS)), iou_vals, marker="o", color=color,
                linewidth=2, label=f"{pair_name} ({cat})")
        for xi, v in enumerate(iou_vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (xi, v),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color=color)

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Density level")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 0.8)
    ax.set_title(f"IoU vs Density Level — {representation} (YZ, Chr1)\n"
                 f"lower levels = dense core, higher levels = global structure")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"iou_vs_level_{representation.lower()}.png"
    out = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


ALL_PAIRS = [
    {"name": "UNTR 12h vs 18h",     "label": "Stability",            "hrs_a": "12hrs", "cond_a": "untr", "hrs_b": "18hrs", "cond_b": "untr"},
    {"name": "VACV 12h vs 18h",     "label": "Temporal change",       "hrs_a": "12hrs", "cond_a": "vacv", "hrs_b": "18hrs", "cond_b": "vacv"},
    {"name": "UNTR vs VACV at 18h", "label": "Condition difference",  "hrs_a": "18hrs", "cond_a": "untr", "hrs_b": "18hrs", "cond_b": "vacv"},
]


def run_all_chroms_all_pairs() -> pd.DataFrame:
    rows = []
    for chrom in CHROMS:
        csvs, labels = [], []
        for hrs in TIMES:
            for cond in CONDS:
                p = os.path.join(DATA_ROOT, chrom, hrs, cond,
                                 f"structure_{hrs}_{cond}_gene_info.csv")
                if os.path.exists(p):
                    csvs.append(p)
                    labels.append(f"{chrom}_{hrs}_{cond}")
        if len(csvs) < 2:
            continue
        try:
            result = mpase.run(
                csv_list=csvs,
                labels=labels,
                xyz_cols=XYZ_COLS,
                cfg_common=CFG_COMMON,
                cfg_hdr=CFG_HDR,
                cfg_pf=CFG_PF,
            )
            for pair in ALL_PAIRS:
                A_label = f"{chrom}_{pair['hrs_a']}_{pair['cond_a']}"
                B_label = f"{chrom}_{pair['hrs_b']}_{pair['cond_b']}"
                for variant, rep_label in [("hdr", "HDR"), ("point_fraction", "PF")]:
                    for level in LEVELS:
                        df_m = result["metrics"]
                        row  = df_m[
                            (df_m["plane"] == PLANE) & (df_m["level"] == level) &
                            (df_m["variant"] == variant) &
                            (df_m["A"] == A_label) & (df_m["B"] == B_label)
                        ]
                        iou = float(row.iloc[0]["IoU"]) if not row.empty else float("nan")
                        rows.append({
                            "chrom":          chrom,
                            "pair":           pair["name"],
                            "category":       pair["label"],
                            "representation": rep_label,
                            "level":          level,
                            "IoU":            round(iou, 4),
                        })
            print(f"  {chrom}: done")
        except Exception as e:
            print(f"  {chrom}: ERROR — {e}")
    return pd.DataFrame(rows)


def plot_iou_vs_level_avg(df: pd.DataFrame, representation: str):
    """
    Line plot with ±std band: mean IoU vs density level across all chromosomes,
    one line per comparison type.
    """
    sub     = df[df["representation"] == representation]
    palette = {"Stability": "#2196F3", "Temporal change": "#FF9800", "Condition difference": "#E53935"}
    pair_order = ["Stability", "Temporal change", "Condition difference"]
    x_pos   = list(range(len(LEVELS)))
    x_labels = [f"{l}%" for l in LEVELS]

    fig, ax = plt.subplots(figsize=(7, 4))
    for cat in pair_order:
        csub = sub[sub["category"] == cat]
        means, stds = [], []
        for level in LEVELS:
            vals = csub[csub["level"] == level]["IoU"].dropna()
            means.append(vals.mean())
            stds.append(vals.std())
        means, stds = np.array(means), np.array(stds)
        color = palette[cat]
        ax.plot(x_pos, means, marker="o", color=color, linewidth=2, label=cat)
        ax.fill_between(x_pos, means - stds, means + stds, color=color, alpha=0.15)
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.annotate(f"{m:.3f}", (xi, m),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Density level")
    ax.set_ylabel("IoU (mean ± std across 24 chromosomes)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"IoU vs Density Level — {representation} (YZ, all chromosomes)\n"
                 f"shaded = ±1 std across chromosomes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"iou_vs_level_avg_{representation.lower()}.png"
    out   = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Running shape representation analysis...")
    df = run_all()

    out_csv = os.path.join(RESULTS_DIR, "shape_representation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    print("\nGenerating figures...")
    plot_grouped_bars(df)
    plot_heatmaps(df)
    plot_delta(df)
    plot_iou_vs_level(df, "HDR")
    plot_iou_vs_level(df, "PF")

    print("\nRunning chromosome-level analysis (UNTR vs VACV at 18h, all chromosomes)...")
    df_chrom = run_chrom_level()
    chrom_csv = os.path.join(RESULTS_DIR, "chrom_level_iou.csv")
    df_chrom.to_csv(chrom_csv, index=False)
    print(f"Saved {chrom_csv}")

    plot_chrom_level_heatmap(df_chrom, "HDR")
    plot_chrom_level_heatmap(df_chrom, "PF")

    print("\nRunning all pairs across all chromosomes (averaged IoU vs level)...")
    df_avg = run_all_chroms_all_pairs()
    avg_csv = os.path.join(RESULTS_DIR, "all_chroms_all_pairs_iou.csv")
    df_avg.to_csv(avg_csv, index=False)
    print(f"Saved {avg_csv}")
    plot_iou_vs_level_avg(df_avg, "HDR")
    plot_iou_vs_level_avg(df_avg, "PF")

    print("\n── Results (60% level) ───────────────────────────────────────")
    print(df[df["level"] == 60].to_string(index=False))


if __name__ == "__main__":
    main()
