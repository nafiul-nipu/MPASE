"""
Generate metrics_data.json, chr1_shape_metrics.csv, and chr1_shape_metrics.xlsx
for chr1 using all 6 conditions (3 timepoints x 2 conditions).

Usage:
    python generate.py
"""

import sys, json
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

import mpase

OUT_DIR   = Path(__file__).parent
DATA_ROOT = _ROOT / "evaluation" / "data" / "all_structure_files"

CHROM    = "chr1"
XYZ_COLS = ("middle_x", "middle_y", "middle_z")
TIMES    = ["12hrs", "18hrs", "24hrs"]
CONDS    = ["untr", "vacv"]

CFG_COMMON = mpase.CfgCommon()
CFG_HDR    = mpase.CfgHDR(n_boot=256)
CFG_PF     = mpase.CfgPF()


def collect(chrom):
    csvs, labels = [], []
    for hrs in TIMES:
        for cond in CONDS:
            p = DATA_ROOT / chrom / hrs / cond / f"structure_{hrs}_{cond}_gene_info.csv"
            if p.exists():
                csvs.append(str(p))
                labels.append(f"{chrom}_{hrs}_{cond}")
    return csvs, labels


def save_json(metrics_df: pd.DataFrame):
    records = []
    for _, row in metrics_df.iterrows():
        records.append({
            "mode":      row["variant"],
            "plane":     row["plane"],
            "level":     int(row["level"]),
            "A":         row["A"],
            "B":         row["B"],
            "IoU":       row["IoU"],
            "meanNN":    row["meanNN"],
            "Hausdorff": row["Hausdorff"],
        })
    out = OUT_DIR / "metrics_data.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {out}  ({len(records)} records)")


def save_csv_xlsx(metrics_df: pd.DataFrame):
    hdr = metrics_df[metrics_df["variant"] == "hdr"]
    PLANE_LEVELS = [("YZ", 60), ("YZ", 95), ("XZ", 60), ("XZ", 95)]

    def cell(A, B, plane, level):
        row = hdr[
            (hdr["A"] == A) & (hdr["B"] == B) &
            (hdr["plane"] == plane) & (hdr["level"] == level)
        ]
        if row.empty:
            return "nan/nan"
        r = row.iloc[0]
        return f"{r['IoU']:.3f}/{r['meanNN']:.2f}"

    rows = []
    time_pairs = [
        ("12hrs", "18hrs", "12h -> 18h"),
        ("12hrs", "24hrs", "12h -> 24h"),
        ("18hrs", "24hrs", "18h -> 24h"),
    ]
    for cond in CONDS:
        for t1, t2, label in time_pairs:
            A = f"{CHROM}_{t1}_{cond}"
            B = f"{CHROM}_{t2}_{cond}"
            r = {"Group": "Within-condition", "Comparison": label, "Condition": cond.upper()}
            for plane, level in PLANE_LEVELS:
                r[f"{plane} ({level}%)"] = cell(A, B, plane, level)
            rows.append(r)

    for hrs in TIMES:
        A = f"{CHROM}_{hrs}_untr"
        B = f"{CHROM}_{hrs}_vacv"
        r = {"Group": "Between-condition", "Comparison": "UNTR vs VACV",
             "Condition": hrs.replace("hrs", "h")}
        for plane, level in PLANE_LEVELS:
            r[f"{plane} ({level}%)"] = cell(A, B, plane, level)
        rows.append(r)

    df = pd.DataFrame(rows)

    csv_out = OUT_DIR / f"{CHROM}_shape_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    xlsx_out = OUT_DIR / f"{CHROM}_shape_metrics.xlsx"
    df.to_excel(xlsx_out, index=False)
    print(f"Saved {xlsx_out}")


def main():
    csvs, labels = collect(CHROM)
    print(f"Running MPASE on {CHROM} ({len(csvs)} CSVs)...")

    result = mpase.run(
        csv_list=csvs,
        labels=labels,
        xyz_cols=XYZ_COLS,
        cfg_common=CFG_COMMON,
        cfg_hdr=CFG_HDR,
        cfg_pf=CFG_PF,
    )

    metrics_df = result["metrics"]
    save_json(metrics_df)
    save_csv_xlsx(metrics_df)
    print("Done.")


if __name__ == "__main__":
    main()
