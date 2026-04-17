# Data

Synthetic point clouds used across all evaluations.

## Plan

### `generate_synthetic.py`

Generates controlled 3D point sets and saves them as `.npy` files under `synthetic/`.

Shapes to generate:
- **S-shape** — two curved lobes, distinctive enough to test alignment and shape extraction
- **Helix** — tests alignment of rotationally symmetric structures
- **Blob** — isotropic Gaussian cloud, baseline/control
- **Two-blob** — two separated clusters, tests multi-component shape handling

Each shape is generated at a fixed seed for reproducibility. The script also produces
**perturbed copies** of each shape (random rotation + translation + gaussian noise) used
by eval1 and eval2.

### `synthetic/`

Output `.npy` files, one per shape/variant:
- `s_shape.npy`, `s_shape_perturbed_{i}.npy`
- `helix.npy`, `helix_perturbed_{i}.npy`
- `blob.npy`, `blob_perturbed_{i}.npy`

## Usage

```bash
python data/generate_synthetic.py
```
