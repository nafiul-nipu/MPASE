# Multi-Point Alignment and Shape Extraction (MPASE)

MPASE is a Python package for **aligning multiple 3D point clouds**, extracting **shape summaries** on shared 2D projection planes, and computing **pairwise comparison metrics** across conditions or time points. It was originally developed for comparative analysis of reconstructed **3D genome structures**, but it is designed to work with **any 3D point sets** (e.g., spatial omics coordinates, particle simulations, tracking data).

This README is written as **package-style documentation** (API + parameters + examples).

---

## Installation

### Option A Install from GitHub (current version)

```bash
pip install git+https://github.com/nafiul-nipu/MPASE.git
```

### Option B (developer / editable install)

```bash
git clone https://github.com/nafiul-nipu/MPASE.git
cd MPASE
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

**Python:** MPASE requires **Python ≥ 3.9**. The package has been tested with 3.9.6 and 3.10.14

---

## Quickstart

### 1) Run MPASE on two CSV files

```python
import mpase

res = mpase.run(
    csv_list=[
        "A.csv",
        "B.csv",
    ],
    labels=["A", "B"],
    xyz_cols=("x","y","z"),
    align_mode="auto",
    run_hdr=True,
    run_pf=True,
    out_dir="mpase_out",
)
print(res["metrics"].head())
```

### 2) Visualize (overlay)

```python
mpase.view(
    res, kind="hdr", plane="XY", levels=[100,95,80],
    A_lab="A", B_lab="B",
    labelA="Condition A", labelB="Condition B",
    show_heat=True,
)
```

### 3) Export a web bundle (D3 / Three.js)

```python
mpase.export_all(
    res,
    out_dir="web_bundle",
    include_density=True,
    kind_levels={"hdr": "all", "point_fraction": "all"},
)
```

---

## Input formats

MPASE accepts input in one of two modes:

### A) CSV mode (`csv_list`)

Each CSV must contain 3 columns for 3D coordinates (default: `middle_x`, `middle_y`, `middle_z`).

Optional: you can provide an ID column (`id_col`) so that MPASE carries per-point IDs through alignment/export.

### B) Array mode (`points_list`)

A list of NumPy arrays of shape `(Ni, 3)` (float32 or float64). Optional per-point IDs can be supplied via `ids_list`.

---

## Core API

### `mpase.run(...)` (alias: `mpase.run_silhouettes(...)`)

```python
mpase.run(
    csv_list=None,
    *,
    points_list=None,
    labels=None,
    xyz_cols=("x","y","z"),
    id_col=None,
    ids_list=None,
    align_mode="auto",
    point_alignment_only=False,
    out_dir=None,
    run_hdr=True,
    run_pf=True,
    cfg_common=None,
    cfg_hdr=None,
    cfg_pf=None,
    planes=("XY","YZ","XZ"),
) -> dict
```

#### Parameters

**Inputs**

- `csv_list` (Sequence[str] | None)  
  List of CSV file paths. Provide **either** `csv_list` **or** `points_list`.
- `points_list` (Sequence[np.ndarray] | None)  
  List of 3D point arrays, each shaped `(Ni, 3)`. Provide **either** `points_list` **or** `csv_list`.
- `labels` (Sequence[str] | None)  
  Names for each input set. If omitted, MPASE auto-generates labels (`S0`, `S1`, ...).  
  Length must match the number of input sets.

**CSV coordinate columns**

- `xyz_cols` (tuple[str,str,str])  
  Column names used in CSV mode. Default: `("x","y","z")`.

**Point IDs (optional but useful for downstream linking)**

- `id_col` (str | None)  
  CSV mode only. Name of a column to use as point IDs (e.g., gene name or stable index).
- `ids_list` (Sequence[Sequence[Any]] | None)  
  Array mode only. A list of per-point ID sequences (one sequence per input set).  
  Each `ids_list[i]` must match `len(points_list[i])`.

**Alignment**

- `align_mode` (`"auto"` | `"skip"`)
  - `"auto"`: center each set, then align all sets to the first set using **PCA pre-alignment + robust ICP**.  
    _Use this when inputs have arbitrary rotation/orientation (common in reconstructed structures)._
  - `"skip"`: center each set only (no cross-set rotational alignment, assuming points are already aligned).  
    _Use this when your data already share a common coordinate frame (e.g., tracking data in world coordinates)._

**Pipeline controls**

- `run_hdr` (bool)  
  If True, compute **HDR** (highest-density-region) silhouettes.
- `run_pf` (bool)  
  If True, compute **point-fraction** silhouettes (top fraction of points by KDE score).
- `point_alignment_only` (bool)  
  If True, stop after alignment + projections + background masks (no silhouettes/metrics).  
  If True, set `out_dir` to write outputs.  
  _Use this to quickly verify alignment quality and projection setup before running HDR/PF (faster iteration)._

**Output**

- `out_dir` (str | None)  
  Directory for writing outputs (metrics CSV, meta JSON, etc.). Some outputs are only written when `out_dir` is provided.
- `planes` (tuple["XY","YZ","XZ"])  
  Which projection planes to compute. Any subset is allowed.

**Configuration**

- `cfg_common` (CfgCommon | None)  
  Controls grid resolution, padding, and ICP behavior.
- `cfg_hdr` (CfgHDR | None)  
  Controls HDR bootstrap density estimation and HDR levels.
- `cfg_pf` (CfgPF | None)  
  Controls KDE bandwidth, fraction levels, raster disk size, and morphology for PF masks.

#### Returns (RunResult-like dict)

The returned object is a Python dict (see `RunResult` in `types.py`) with the following keys:

- `labels`: list[str]
- `aligned_points`: list[np.ndarray]  
  A list of aligned 3D point arrays, each shaped `(Ni, 3)`.
- `shapes`: dict  
  Shape products for each `variant` (`"hdr"`, `"point_fraction"`), each `plane`, each `level`, and each `label`.
- `metrics`: pandas.DataFrame  
  Pairwise metrics (IoU, meanNN, Hausdorff) computed per plane/level/variant.
- `meta`: dict  
  Metadata including available planes/levels/labels and grid parameters.
- `background`: dict[plane -> bool mask]  
  Union-of-presence mask per plane.  
  _Used as a shared spatial frame so contours/densities from different labels are comparable on the same grid._
- `densities`: dict[label -> dict[plane -> density]] | None  
  Present only if HDR was computed.
- `projections`: dict  
  2D projections per plane, including grid centers (`xs`,`ys`) and 2D point sets per label.

---

## Configuration objects (dataclasses)

These are user-facing dataclasses you can pass to `mpase.run(...)`.

### `mpase.CfgCommon(grid_base=160, pad_frac=0.05, trim_q=0.10, icp_iters=30, sample_icp=50000)`

Controls shared grid construction and alignment behavior.

#### Parameters

- `grid_base` (int, default=160)  
  The base resolution of the shared 2D projection grid. Larger values produce higher-resolution masks/densities but use more memory/time.  
  _Increase for smoother contours; decrease for faster runs / lower memory._
- `pad_frac` (float, default=0.05)  
  Padding around the global bounding box (fraction of max extent) before building the shared grid.  
  _Prevents shapes near the boundary from being clipped after alignment._
- `trim_q` (float, default=0.10)  
  Robust ICP trimming fraction per iteration (discard the worst `trim_q` matches). Range: `[0, 0.5)`
  0.0 = use all points.  
  _Higher values ignore more outliers (good for noisy reconstructions) but may underfit if data are already clean._
- `icp_iters` (int, default=30)  
  Number of ICP iterations. Increase (e.g., 50) for hard alignment cases.
- `sample_icp` (int, default=50000)  
  Maximum number of points sampled per set for ICP. Use `None` to use all points.

---

### `mpase.CfgHDR(n_boot=256, sample_frac=1.0, sigma_px=1.2, density_floor_frac=0.002, mass_levels=(...), rng_seed=0)`

Controls bootstrap-averaged 2D densities and HDR silhouette extraction.

#### Parameters

- `n_boot` (int, default=256)  
  Number of bootstrap resamples per set.  
  _More bootstraps reduce sampling noise in the density estimate but increase runtime._
- `sample_frac` (float, default=1.0)  
  Fraction of points per resample. `1.0` means sampling with replacement (standard bootstrap).
- `sigma_px` (float, default=1.2)  
  Gaussian smoothing sigma (in pixels) applied to the 2D histogram density.  
  _Higher values smooth the density more (cleaner shapes) but can blur fine structure._
- `density_floor_frac` (float, default=0.002)  
  Zeros out tiny densities before HDR threshold search (fraction of `D.max()`).
- `mass_levels` (tuple[float,...])  
  HDR coverage levels in `[0,1]` (e.g., 0.95 means the smallest region containing 95% of probability mass).  
  MPASE converts these to percentage levels (e.g., 95) for indexing/plotting.
- `rng_seed` (int, default=0)  
  Random seed for reproducible bootstrap results.

---

### `mpase.CfgMorph(closing=1, opening=1, keep_largest=False, fill_holes=True)`

Optional morphological cleanup applied to **point-fraction masks**.

#### Parameters

- `closing` (int, default=1)  
  Number of binary closing iterations (seals small gaps).
- `opening` (int, default=1)  
  Number of binary opening iterations (removes isolated specks).
- `keep_largest` (bool, default=False)  
  If True, keep only the largest connected component.
- `fill_holes` (bool, default=True)  
  If True, fill holes in the kept component(s).

---

### `mpase.CfgPF(frac_levels=(...), bandwidth=None, disk_px=2, morph=CfgMorph(...))`

Controls point-fraction silhouette extraction.

#### Parameters

- `frac_levels` (tuple[float,...])  
  Fraction levels in `[0,1]`. For each `f`, MPASE keeps the top `ceil(f*N)` points by KDE score before rasterizing. 1 means include 100% or all points.  
  _Lower fractions highlight dense cores; higher fractions include broader structure._
- `bandwidth` (float | None, default=None)  
  KDE bandwidth. If None, MPASE selects an automatic bandwidth based on point spacing.  
  _Controls how local vs global the KDE scoring is (smaller = more local detail; larger = smoother ranking)._
- `disk_px` (int, default=2)  
  Raster disk radius (in pixels) when painting kept points into the mask grid.  
  _Larger disks make masks more continuous for sparse data but can over-thicken shapes._
- `morph` (CfgMorph)  
  Morphological cleanup configuration applied after rasterization.

---

## Visualization API

Visualization functions operate on the result returned by `mpase.run(...)`.

### `mpase.view(...)` (overlay: compare two labels)

```python
mpase.view(
    result,
    kind="hdr",
    plane="XY",
    levels="all",
    *,
    A_lab=None,
    B_lab=None,
    labelA="A",
    labelB="B",
    show_heat=False,
    cfg_hdr=None,
    cfg_pf=None,
    clean_blobs=False,
    blob_min_len=10,
    blob_min_area_frac=0.05,
)
```

#### Key parameters

- `kind` (`"hdr"` | `"point_fraction"`)  
  Which silhouette variant to display.
- `plane` (`"XY" | "YZ" | "XZ"`)  
  Projection plane to display.
- `levels` (`"all"` | int | list[int])  
  Levels to show. Levels are in **percent** (e.g., 100, 95, 80).  
  _A common choice is [100, 80] to compare the main body and the dense core without showing every level._
- `A_lab`, `B_lab` (str | None)  
  Which two labels to overlay. If None, MPASE uses the first two labels in `result["labels"]`.
- `show_heat` (bool)  
  If True and `kind="hdr"` and densities exist, show the density heatmap underlay (for `A_lab`).
- `clean_blobs`, `blob_min_len`, `blob_min_area_frac`  
  Optional contour cleanup: remove tiny/noisy contours before plotting.  
  _Enable this when contours look speckled due to sparse points or grid discretization; it does not change the underlying masks/densities._

---

### `mpase.save_figures(...)` (overlay → PNG)

#### Key parameters

Same signature as `view`, plus:

- `out_dir` (str, default="figures")  
  Output folder for PNGs.

---

### `mpase.view_projections(...)` (aligned points scatter)

```python
mpase.view_projections(
    result,
    *,
    planes=("XY","YZ","XZ"),
    A_lab=None,
    B_lab=None,
    labelA="A",
    labelB="B",
    s=3.0,
    alphaA=0.7,
    alphaB=0.7,
)
```

Displays 2D scatter projections for the chosen planes.

#### Key parameters

- `result` (`dict`)  
  Output returned by `mpase.run(...)`.

- `planes` (`tuple[str]`)  
  Projection planes to display (default: `("XY", "YZ", "XZ")`).

- `A_lab`, `B_lab` (`str | None`)  
  Labels of the two point sets to display. If `None`, the first two labels in `result["labels"]` are used.

- `labelA`, `labelB` (`str`)  
  Display names used in the legend for datasets `A_lab` and `B_lab`.

- `s` (`float`)  
  Marker size for scatter points (passed to Matplotlib).  
  Smaller values are recommended for dense point clouds.

- `alphaA`, `alphaB` (`float`)  
  Transparency values for scatter points of datasets `A_lab` and `B_lab`.  
  Values range from `0` (fully transparent) to `1` (fully opaque).

---

### `mpase.save_projections(...)` (scatter → PNG, optional CSV)

```python
mpase.save_projections(
    result,
    *,
    out_dir="figures",
    planes=("XY","YZ","XZ"),
    A_lab=None,
    B_lab=None,
    labelA="A",
    labelB="B",
    s=3.0,
    alphaA=0.7,
    alphaB=0.7,
    dpi=220,
    save_csv=False,
)
```

#### Key parameters

Same signature as `view_projections`, plus:

- `out_dir` (`str`)  
  Directory where projection figures (and optional CSV files) are saved.

- `dpi` (`int`)  
  Resolution (dots per inch) of the saved PNG figures.

- `save_csv` (`bool`)  
  If `True`, also export the projected 2D point coordinates as CSV files for each plane and label.

---

### `mpase.view_single(...)` (single label, no overlay)

```python
mpase.view_single(
    result,
    label,
    kind="hdr",
    plane="XY",
    levels="all",
    *,
    show_heat=False,
    cfg_hdr=None,
    cfg_pf=None,
    clean_blobs=False,
    blob_min_len=10,
    blob_min_area_frac=0.05,
)
```

Shows silhouettes for **one label** at a time. If per-label backgrounds exist, they are used automatically.

#### Key parameters

- `result` (`dict`)  
  Output returned by `mpase.run(...)`.

- `label` (`str`)  
  Label of the dataset to visualize.

- `kind` (`"hdr"` | `"point_fraction"`)  
  Which silhouette variant to display.

- `plane` (`"XY"` | `"YZ"` | `"XZ"`)  
  Projection plane to display.

- `levels` (`"all"` | `int` | `list[int]`)  
  Levels to show (percent values such as `100`, `95`, `80`).

- `show_heat` (`bool`)  
  If `True` and `kind="hdr"` and densities exist, show the density heatmap underlay.

- `cfg_hdr`, `cfg_pf` (`CfgHDR | None`, `CfgPF | None`)  
  Optional configuration overrides for visualization.

- `clean_blobs` (`bool`)  
  If `True`, remove small or noisy contour fragments before plotting.

- `blob_min_len` (`int`)  
  Minimum contour length (in pixels) to keep when cleaning blobs.

- `blob_min_area_frac` (`float`)  
  Minimum contour area as a fraction of total mask area to keep.

---

### `mpase.save_per_label(...)` (single label → PNG)

```python
mpase.save_per_label(
    result,
    labels=None,
    *,
    kind="hdr",
    plane="XY",
    levels="all",
    out_dir="figures_single",
    show_heat=False,
    cfg_hdr=None,
    cfg_pf=None,
    dpi=220,
    clean_blobs=False,
    blob_min_len=10,
    blob_min_area_frac=0.05,
)
```

Saves `label × level` PNGs without overlay.

#### Key parameters

- `result` (`dict`)  
  Output returned by `mpase.run(...)`.

- `labels` (`list[str] | None`)  
  Labels to export. If `None`, all labels in `result["labels"]` are used.

- `kind` (`"hdr"` | `"point_fraction"`)  
  Which silhouette variant to save.

- `plane` (`"XY"` | `"YZ"` | `"XZ"`)  
  Projection plane to export.

- `levels` (`"all"` | `int` | `list[int]`)  
  Levels to save (percent values).

- `out_dir` (`str`)  
  Directory where PNG figures are written.

- `show_heat` (`bool`)  
  If `True` and `kind="hdr"` and densities exist, include the density heatmap underlay.

- `cfg_hdr`, `cfg_pf` (`CfgHDR | None`, `CfgPF | None`)  
  Optional configuration overrides for figure generation.

- `dpi` (`int`)  
  Resolution (dots per inch) of the saved PNG figures.

- `clean_blobs`, `blob_min_len`, `blob_min_area_frac`  
  Optional contour cleanup parameters (same behavior as `view_single`).

---

## Export API (for web visualization)

### `mpase.export_all(...)`

```python
mpase.export_all(
    result,
    out_dir="web_data",
    *,
    include_density=True,
    export_layout=True,
    export_scales=True,
    kind_levels={"hdr": "all", "point_fraction": "all"},
    which_density=None,
    progress_report=False,
    clean_blobs=False,
    blob_min_len=10,
    blob_min_area_frac=0.05,
) -> None
```

#### Parameters

- `out_dir` (str)  
  Output directory for the exported bundle.
- `include_density` (bool)  
  If True and HDR ran, export per-label density fields.
- `export_layout` (bool)  
  If True, export `layout.json` for 3D slice placement (used by some frontends).
- `export_scales` (bool)  
  If True, export `scales.json` containing global bbox min/max across all aligned points.
- `kind_levels` (dict)  
  Which levels to export, per variant. Default `all` for HDR and PF Example:
  ```python
  {"hdr": [100,95,80], "point_fraction": "all"}
  ```
- `which_density` (Iterable[str] | None)  
  Restrict density export to a subset of labels.
- `progress_report` (bool)  
  If True, print progress notifications.
- `clean_blobs`, `blob_min_len`, `blob_min_area_frac`  
  Optional contour cleanup before exporting contours.

#### Outputs

Most web-based visualization frontends typically use the contour data, 2D projections,
3D point coordinates, and metadata. Density files are optional and are only required
when rendering heatmaps or density underlays.

`export_all` writes a manifest plus structured JSON files under `out_dir`, including:

- `meta_data.json`
- `background_mask.json`
- `background_mask_by_label.json` (if available)
- `contours/`
- `density/` (optional)
- `projections/`
- `metrics_data.json`
- `points3d/`
- `layout.json` (optional)
- `scales.json` (optional)
- `manifest.json`

---

## Helper API

### `mpase.load_points(csv, cols=("x","y","z")) -> np.ndarray`

Loads a single CSV into a `(N,3)` float32 array, with basic column checking and `dropna()`.

Example:

```python
from mpase import load_points
P = load_points("A.csv", cols=("x","y","z"))
```

---

## Common usage patterns

### Compare more than two conditions/time points

MPASE supports N sets. Metrics are computed pairwise across labels.

```python
res = mpase.run(
    csv_list=["A.csv","B.csv","C.csv"],
    labels=["A","B","C"],
    run_hdr=True,
    run_pf=False,
)
```

Then visualize any pair:

```python
mpase.view(res, kind="hdr", plane="XY", levels=[95], A_lab="A", B_lab="C")
```

### Alignment-only mode (fast)

```python
res = mpase.run(
    csv_list=["A.csv","B.csv"],
    labels=["A","B"],
    align_mode="auto",
    point_alignment_only=True,
    out_dir="align_only_out",
    run_hdr=False,
    run_pf=False,
)
```

---

## Troubleshooting

### “It runs locally but fails after `pip install git+...`”

- Ensure your package is in `src/mpase/` (not `mpase/` at repo root).
- Ensure `pyproject.toml` includes:
  - `package-dir = {"" = "src"}`
  - a packages find section pointing at `src`.

### “My results changed after edits”

If you changed code, ensure you are running editable install:

```bash
pip install -e .
```

Then verify import path:

```bash
python -c "import mpase; print(mpase.__file__)"
```

---

## Citation / attribution

If you use MPASE in academic work, please cite the associated paper (to be added after publication).

---

## License

MIT.
