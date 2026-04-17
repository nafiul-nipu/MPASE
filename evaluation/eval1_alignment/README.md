# Eval 1 — Alignment Validation

**Goal:** Show that MPASE's alignment (PCA + robust ICP) recovers the correct transformation
when given a known perturbation.

## Method

1. Take a synthetic point cloud (S-shape, helix, blob) as the reference.
2. Apply a known random rigid transformation (rotation R, translation t) + optional Gaussian noise.
3. Run `mpase.align_points()` on the original + perturbed pair.
4. Measure alignment error as RMSE between the aligned perturbed set and the original.
5. Repeat across N perturbations and noise levels.

## Variables

- Shape: S-shape, helix, blob
- Noise level: 0%, 1%, 5%, 10% of point cloud extent
- N perturbations per condition: 20

## Key metric

**RMSE** — root mean squared distance between aligned points and ground truth.
Lower is better. Compare against: no-alignment baseline (RMSE of unaligned perturbed set).

## Expected output

`results/rmse_results.csv` — columns: `shape`, `noise_level`, `trial`, `rmse_before`, `rmse_after`

`figures/rmse_by_noise.png` — RMSE before vs after alignment, grouped by noise level.
