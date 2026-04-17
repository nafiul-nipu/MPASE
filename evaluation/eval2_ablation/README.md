# Eval 2 — Ablation Study

**Goal:** Show that each component of the MPASE alignment pipeline contributes meaningfully.
PCA+ICP should outperform PCA-only, which should outperform no alignment.

## Method

1. Take a synthetic point cloud pair (original + perturbed).
2. Run MPASE three times with different alignment strategies:
   - **No alignment** (`align_mode="skip"`, no centering)
   - **PCA only** (run PCA pre-alignment, skip ICP — requires internal flag or manual implementation)
   - **PCA + ICP** (`align_mode="auto"` — the full pipeline)
3. For each strategy, extract HDR and PF shapes at multiple levels.
4. Compute IoU and Hausdorff between the aligned pair's shapes.

## Variables

- Shape: S-shape, helix, blob
- Level: 100%, 95%, 80%, 60%
- N perturbations per condition: 20

## Key metrics

- **IoU** — higher is better (shapes overlap more after better alignment)
- **Hausdorff distance** — lower is better (contours closer after better alignment)

## Expected output

`results/ablation_results.csv` — columns: `shape`, `align_mode`, `level`, `plane`, `trial`, `IoU`, `Hausdorff`

`figures/iou_by_method.png` — IoU boxplot for each alignment method.
`figures/hausdorff_by_method.png` — Hausdorff boxplot for each alignment method.
