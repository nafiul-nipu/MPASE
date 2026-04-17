# Eval 3 — Stability (Bootstrap Robustness)

**Goal:** Show that MPASE's HDR shape estimates are stable across different bootstrap
random seeds — i.e., the silhouettes are not sensitive to sampling randomness.

## Method

1. Take a fixed synthetic point cloud pair (no perturbation).
2. Run `mpase.run()` with HDR enabled, varying only `CfgHDR(rng_seed=i)` across N runs.
3. For each run, extract HDR masks at each level.
4. Compute pairwise IoU between all run pairs at each level.
5. Report mean IoU and standard deviation across runs.

High mean IoU + low std = stable/robust shapes.

## Variables

- Shape: S-shape, helix
- N seeds: 30
- Levels: 100%, 95%, 80%, 60%
- Planes: XY, YZ, XZ

## Key metric

**IoU variance across seeds** — lower variance = more stable.
Also report: mean IoU (should be high, near 1.0 for stable shapes).

## Expected output

`results/stability_results.csv` — columns: `shape`, `plane`, `level`, `seed_i`, `seed_j`, `IoU`

`figures/iou_variance_by_level.png` — IoU std dev per level, showing stability across bootstrap seeds.
`figures/iou_heatmap.png` — pairwise IoU matrix across seeds for a representative level.
