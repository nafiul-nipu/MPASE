# MPASE Evaluation

Quantitative evaluations for the MPASE paper. Each subfolder is a self-contained experiment
that reads from `data/` and writes results as CSVs + figures.

## Structure

| Folder | Evaluation | Key metric |
|---|---|---|
| `eval1_alignment/` | Alignment validation — known perturbations | RMSE |
| `eval2_ablation/` | Ablation — no-align vs PCA vs PCA+ICP | IoU, Hausdorff |
| `eval3_stability/` | Stability — bootstrap variance | IoU std dev |
| `eval4_multichrom/` | Multi-chromosome generalization (optional) | IoU, Hausdorff |

## How to run

Run each eval independently:
```bash
python eval1_alignment/run.py
python eval2_ablation/run.py
python eval3_stability/run.py
python eval4_multichrom/run.py
```

Then generate all paper figures:
```bash
python summary_figures.py
```

## Dependencies

```bash
pip install -r requirements.txt
```
