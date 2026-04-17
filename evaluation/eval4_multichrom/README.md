# Eval 4 — Multi-Chromosome Generalization (Optional)

**Goal:** Show that MPASE generalizes across real biological data from multiple chromosomes,
not just the development dataset.

## Method

1. Load real 3D genome structure data for multiple chromosomes (e.g., chr1, chr2, chr5, chr10).
2. For each chromosome, run `mpase.run()` on two or more conditions (e.g., treated vs untreated).
3. Collect IoU and Hausdorff metrics per chromosome, plane, and level.
4. Summarize across chromosomes to show consistent performance.

## Variables

- Chromosomes: TBD (depends on available data)
- Conditions: at least 2 per chromosome
- Levels: 100%, 95%, 80%

## Key metrics

- **IoU per chromosome** — consistency across chromosomes
- **Hausdorff per chromosome** — shape boundary agreement

## Expected output

`results/multichrom_results.csv` — columns: `chrom`, `plane`, `level`, `variant`, `IoU`, `Hausdorff`

`figures/iou_by_chrom.png` — IoU across chromosomes, grouped by level.
`figures/summary_heatmap.png` — IoU heatmap: chromosomes × levels.

## Notes

- Real data paths will be added once data is available.
- This eval is optional for the paper but strengthens the generalization claim.
