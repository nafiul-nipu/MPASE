import os
import re
import json
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional, List, Any, Callable
import numpy as np

# Internal types and configs (from the main pipeline module)
from .types import RunResult, Plane, CfgHDR, CfgPF, ShapeProduct
from .visualization_save_image import _levels_from_result
from .metrics_calculation import all_contours_from_bool
from .create_grid_planes import points_to_pixel_indices

##################### PURE-DATA EXPORTERS (D3 + THREE) #####################
# background mask
# contours (per-label files)
# densities (per-label files)
# projections (per-plane files)
# points3D (per-label files)
# scales (bbox, mins)
# meta_data.json
# layout
# metrics_data.json
# + Progress / reporting + manifest

################# Helpers #################

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _write_json(path: str, obj) -> str:
    """Write JSON and return the path (for manifest)."""
    with open(path, "w") as f:
        # Pretty for readability; small files remain fine, large files will still be large.
        json.dump(obj, f, separators=(",", ":"), allow_nan=False, indent=2)
    return path


def _grid_sizes_from_result(result: RunResult) -> Dict[Plane, Tuple[int, int]]:
    # nx,ny from background masks (bool arrays are [ny, nx])
    return {plane: (bg.shape[1], bg.shape[0]) for plane, bg in result["background"].items()}


def _safe_name(s: str) -> str:
    """Make a filesystem-safe, compact filename from a label or plane name."""
    s = str(s).strip()
    # Replace spaces, slashes, and other non-word chars with underscore; collapse repeats.
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "unnamed"


def _notify(progress_report: bool, event: str, **payload):
    if progress_report:
        msg = f"[export] {event}"
        if payload:
            msg += ": " + ", ".join(f"{k}={v}" for k, v in payload.items() if k != "obj")
        print(msg)


################# Core #################

def export_meta(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    meta_data.json:
      {
        "planes": ["XY","YZ","XZ"],
        "grid": {"XY":[nx,ny], ...},
        "levels": {"hdr":[...], "pf":[...]},
        "labels": ["UNTR","VACV", ...]
      }
    """
    _ensure_dir(out_dir)
    planes = list(result["background"].keys())
    grid = {p: list(_grid_sizes_from_result(result)[p]) for p in planes}
    labels = list(result.get("labels", []))

    levels_hdr, levels_pf = set(), set()
    if "hdr" in result["shapes"]:
        for plane in result["shapes"]["hdr"].keys():
            levels_hdr.update(result["shapes"]["hdr"][plane].keys())
    if "point_fraction" in result["shapes"]:
        for plane in result["shapes"]["point_fraction"].keys():
            levels_pf.update(result["shapes"]["point_fraction"][plane].keys())

    meta = dict(
        planes=planes,
        grid=grid,
        levels={
            "hdr": sorted([int(x) for x in levels_hdr], reverse=True),
            "pf": sorted([int(x) for x in levels_pf], reverse=True),
        },
        labels=labels,
    )
    path = _write_json(os.path.join(out_dir, "meta_data.json"), meta)
    _notify(progress_report, "write", kind="meta", path=path)
    return [path]


################### D3: background-as-data ###################

def export_background_mask_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    _ensure_dir(out_dir)
    bg = {}
    for plane, arr in result["background"].items():
        # store as 0/1 (compact)
        bg[plane] = arr.astype(np.uint8).tolist()
    path = _write_json(os.path.join(out_dir, "background_mask.json"), bg)
    _notify(progress_report, "write", kind="background_mask", path=path)
    return [path]


################### D3: background masks per label ###################

def export_background_mask_by_label_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    Writes one JSON per label under <out_dir>/background_by_label/ with per-plane 0/1 masks.
    """
    written: List[str] = []
    if "background_by_label" not in result:
        return written
    bgdir = os.path.join(out_dir, "background_by_label")
    _ensure_dir(bgdir)
    by_plane = result["background_by_label"]
    # Collect labels
    labels = list(result.get("labels", []))
    for lab in labels:
        payload = {}
        for plane, d in by_plane.items():
            if lab in d:
                payload[plane] = d[lab].astype(np.uint8).tolist()
        if not payload:
            continue
        fname = f"{_safe_name(lab)}_background.json"
        path = _write_json(os.path.join(bgdir, fname), payload)
        written.append(path)
        _notify(progress_report, "write", kind="background_by_label", label=str(lab), path=path)
    return written


################### D3: densities (per label) ###################

def export_density_json(result: RunResult, out_dir: str, which: Optional[Iterable[str]] = None, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    Writes density fields as nested lists (floats) per label. Only if HDR was run.
    Output directory: <out_dir>/density/
    Files: <Label>_density.json (e.g., 12h_UNTR_density.json)
    """
    written: List[str] = []
    if not result.get("densities"):
        return written
    den_dir = os.path.join(out_dir, "density")
    _ensure_dir(den_dir)
    labels = list(which) if which is not None else list(result["densities"].keys())
    for lab in labels:
        if lab not in result["densities"]:
            continue
        payload = {plane: D.astype(float).tolist() for plane, D in result["densities"][lab].items()}
        fname = f"{_safe_name(lab)}_density.json"
        path = _write_json(os.path.join(den_dir, fname), payload)
        written.append(path)
        _notify(progress_report, "write", kind="density", label=str(lab), path=path)
    return written


################### D3: contours (pixel coords) ###################

def export_contours_d3(
    result: RunResult,
    out_dir: str,
    kind_levels: Dict[str, "int|Iterable[int]|str"] = {"hdr": "all", "point_fraction": "all"},
    *,
    clean_blobs: bool = False,
    blob_min_len: int = 10,
    blob_min_area_frac: float = 0.05,
    progress_report: bool = False,
    report: Optional[Callable] = None,
) -> List[str]:
    """
    Writes per-label contour bundles under <out_dir>/contours/.
    Each file aggregates that label's contours across planes and levels.

    contours/<Label>_contour.json
      {
        "contours": [
          {"plane": "XY", "variant": "hdr"|"pf", "level": 95,
           "label": "12h_UNTR", "points": [[x,y], ...]},
          ...
        ]
      }

    Now: for each ShapeProduct, we may emit MULTIPLE entries (one per blob),
    using all_contours_from_bool(shape['mask'], ...).
    """
    written: List[str] = []
    cont_dir = os.path.join(out_dir, "contour")
    _ensure_dir(cont_dir)

    per_label: Dict[str, list] = {}

    cfg_hdr = CfgHDR()
    cfg_pf = CfgPF()

    for kind, lv in kind_levels.items():
        if kind not in result["shapes"]:
            continue
        levels = _levels_from_result(kind, cfg_hdr, cfg_pf, lv)

        for plane, by_level in result["shapes"][kind].items():
            for level in levels:
                by_label = by_level.get(level)
                if not by_label:
                    continue

                for label, sp in by_label.items():
                    mask = sp.get("mask")
                    if mask is None:
                        continue

                    # get ALL blobs for this mask
                    contours = all_contours_from_bool(mask, min_len=blob_min_len, min_area_frac=blob_min_area_frac if clean_blobs else 0.0)
                    if not contours:
                        continue

                    for C in contours:
                        # skimage contours are [row(y), col(x)] — convert to [x,y]
                        pts = [[float(c[1]), float(c[0])] for c in C]
                        entry = dict(
                            plane=plane,
                            variant="hdr" if kind == "hdr" else "pf",
                            level=int(level),
                            label=str(label),
                            points=pts,
                        )
                        per_label.setdefault(str(label), []).append(entry)

    # Write one file per label
    for label, entries in per_label.items():
        fname = f"{_safe_name(label)}_contour.json"
        path = _write_json(os.path.join(cont_dir, fname), {"contours": entries})
        written.append(path)
        _notify(
            progress_report,
            "write",
            kind="contours",
            label=str(label),
            path=path,
            entries=len(entries),
        )

    return written



################### D3: 2D projections (per plane) ###################

def export_projections_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    Writes one file per plane under <out_dir>/projections/ in *pixel* grid coords that match contours:
    projections/<PLANE>_projections.json
      {"UNTR": [[x,y], ...], "VACV": [[x,y], ...], ...}

    Uses result["projections"][plane]["sets"][label] as 2D points.
    """
    written: List[str] = []
    proj_dir = os.path.join(out_dir, "projections")
    _ensure_dir(proj_dir)

    # grid sizes from background (to align with contour pixel space)
    grid = {plane: (bg.shape[1], bg.shape[0]) for plane, bg in result["background"].items()}  # (nx, ny)

    for plane, d in result.get("projections", {}).items():
        if plane not in grid:
            continue
        nx, ny = grid[plane]
        sets2d = d.get("sets", {})
        plane_out = {}

        def to_pixel(points: np.ndarray):
            if points.size == 0:
                return points.tolist()
            P = points.astype(float).copy()

            # If normalized 0..1 → scale to pixel grid
            if np.all((P >= 0) & (P <= 1)):
                P[:, 0] *= (nx - 1)
                P[:, 1] *= (ny - 1)

            # Flip Y to image coordinates (origin top-left)
            P[:, 1] = (ny - 1) - P[:, 1]
            return P.tolist()

        for lab, P2 in sets2d.items():
            plane_out[str(lab)] = to_pixel(np.asarray(P2))

        fname = f"{_safe_name(plane)}_projections.json"
        path = _write_json(os.path.join(proj_dir, fname), plane_out)
        written.append(path)
        _notify(progress_report, "write", kind="projections", plane=str(plane), path=path, labels=len(plane_out))

    return written


############## Metrics ################

def export_metrics_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    metrics_data.json: list of rows; already pairwise (A,B) in your DataFrame.
    """
    _ensure_dir(out_dir)
    rows = result["metrics"].to_dict(orient="records")
    # ensure json-serializable types
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v)
    path = _write_json(os.path.join(out_dir, "metrics_data.json"), rows)
    _notify(progress_report, "write", kind="metrics", path=path, rows=len(rows))
    return [path]


#################### Three.js: 3D points (per label) ###################

def export_points3d_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    Writes one file per label under <out_dir>/points3d/:
    points3d/<Label>_points3d.json
      {"positions": [[x,y,z], ...]}

    Uses result["aligned_points"] (list aligned to result["labels"]).
    """
    written: List[str] = []
    pts_dir = os.path.join(out_dir, "points3d")
    _ensure_dir(pts_dir)
    labels = list(result.get("labels", []))
    aligned = list(result.get("aligned_points", []))

    for lab, X in zip(labels, aligned):
        payload = {"positions": np.asarray(X, dtype=float).tolist()}
        fname = f"{_safe_name(lab)}_points3d.json"
        path = _write_json(os.path.join(pts_dir, fname), payload)
        written.append(path)
        _notify(progress_report, "write", kind="points3d", label=str(lab), path=path, count=len(payload["positions"]))

    return written


################# Optional: scene layout and scales #################

def export_layout_json(out_dir: str, layout: Optional[Dict[str, Dict[str, list]]] = None, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    layout.json:
      { "XY":{"origin":[0,0,0],"normal":[0,0,1]}, ... }
    """
    _ensure_dir(out_dir)
    layout = layout or {
        "XY": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
        "YZ": {"origin": [1.2, 0, 0], "normal": [1, 0, 0]},
        "XZ": {"origin": [0, -1.2, 0], "normal": [0, 1, 0]},
    }
    path = _write_json(os.path.join(out_dir, "layout.json"), layout)
    _notify(progress_report, "write", kind="layout", path=path)
    return [path]


def export_scales_json(result: RunResult, out_dir: str, *, progress_report: bool = False, report: Optional[Callable] = None) -> List[str]:
    """
    scales.json: bounding box across ALL aligned points.
    """
    _ensure_dir(out_dir)
    if not result.get("aligned_points"):
        path = _write_json(os.path.join(out_dir, "scales.json"), {"bbox": {"mins": [0, 0, 0], "maxs": [0, 0, 0]}})
        _notify(progress_report, "write", kind="scales", path=path)
        return [path]
    P = np.vstack([np.asarray(X) for X in result["aligned_points"] if X is not None and len(X) > 0])
    mins = P.min(axis=0).astype(float).tolist()
    maxs = P.max(axis=0).astype(float).tolist()
    path = _write_json(os.path.join(out_dir, "scales.json"), {"bbox": {"mins": mins, "maxs": maxs}})
    _notify(progress_report, "write", kind="scales", path=path, mins=mins, maxs=maxs)
    return [path]

def export_membership_json(
    result: RunResult,
    out_dir: str,
    *,
    progress_report: bool = False,
    report: Optional[Callable] = None,
) -> List[str]:
    """
    membership.json:
    {
      "<label>": {
        "points": N,
        "ids": [...],
        "planes": {
          "XY": {
            "pixels": [[x,y], ...],             # per-gene pixel coords (mask space)
            "hdr": { "100": [0,1,5,...], ... }, # gene indices inside each shape
            "point_fraction": { ... }
          },
          ...
        }
      },
      ...
    }
    """
    _ensure_dir(out_dir)
    membership = {}
    labels = list(result.get("labels", []))
    proj = result.get("projections", {})
    shapes = result.get("shapes", {})
    ids_by_label = result.get("ids_by_label", {})

    for lab in labels:
        lab_entry = {
            "points": 0,
            "ids": ids_by_label.get(lab, []),
            "planes": {}
        }

        # assume all planes share same N for this label
        # pick any plane in projections
        some_plane = next(iter(proj.keys()))
        N = proj[some_plane]["sets"][lab].shape[0]
        lab_entry["points"] = int(N)

        for plane, pdata in proj.items():
            sets2d = pdata["sets"]
            if lab not in sets2d:
                continue

            P2 = np.asarray(sets2d[lab])
            xs, ys = pdata["xs"], pdata["ys"]

            # per-gene pixel indices in THIS plane
            x_idx, y_idx = points_to_pixel_indices(P2, xs, ys)
            pixels = np.stack([x_idx, y_idx], axis=1).astype(float).tolist()

            plane_entry = {
                "pixels": pixels,         # <<< new
                "hdr": {},
                "point_fraction": {},
            }

            # for each variant/level, compute membership by sampling mask[y_idx, x_idx]
            for variant in ("hdr", "point_fraction"):
                if variant not in shapes:
                    continue
                if plane not in shapes[variant]:
                    continue
                for level, by_label in shapes[variant][plane].items():
                    sp = by_label.get(lab)
                    if not sp:
                        continue
                    mask = sp.get("mask")
                    if mask is None:
                        continue

                    inside = mask[y_idx, x_idx]  # bool [N]
                    idxs = np.nonzero(inside)[0].astype(int).tolist()
                    if idxs:
                        plane_entry[variant][str(level)] = idxs

            lab_entry["planes"][plane] = plane_entry

        membership[lab] = lab_entry

    path = _write_json(os.path.join(out_dir, "membership.json"), membership)
    _notify(progress_report, "write", kind="membership", path=path)
    return [path]


################ One-call convenience ################

def export_all(
    result: RunResult,
    out_dir: str = "web_data",
    *,
    include_density: bool = True,
    export_layout: bool = True,
    export_scales: bool = True,
    kind_levels: Dict[str, "int|Iterable[int]|str"] = {"hdr": "all", "point_fraction": "all"},
    which_density: Optional[Iterable[str]] = None,
    progress_report: bool = False,
    clean_blobs: bool = False,
    blob_min_len: int = 10,
    blob_min_area_frac: float = 0.05,
) -> None:
    """
    Produces the full pure-data bundle for D3 + Three.js and writes a manifest file (manifest.json):
      - meta_data.json (root)
      - background_mask.json (root)
      - contours/<Label>_contour.json (per label)
      - (opt) density/<Label>_density.json (per label, if HDR available)
      - projections/<PLANE>_projections.json (per plane)
      - metrics_data.json (root)
      - points3d/<Label>_points3d.json (per label)
      - (opt) layout.json (root)
      - (opt) scales.json (root)

    New:
      - progress notifications via `progress_report` prints
      - writes manifest.json summarizing outputs
    """
    _ensure_dir(out_dir)
    _notify(progress_report, "begin", out_dir=out_dir)

    manifest: Dict[str, Any] = {"root": out_dir, "written": {}}

    def rec(name: str, paths: List[str]):
        manifest["written"].setdefault(name, []).extend(paths)

    rec("meta", export_meta(result, out_dir, progress_report=progress_report))
    rec("background", export_background_mask_json(result, out_dir, progress_report=progress_report))
    rec("background_by_label", export_background_mask_by_label_json(result, out_dir, progress_report=progress_report))
    if include_density:
        rec("density", export_density_json(result, out_dir, which=which_density, progress_report=progress_report))
    rec("contours", export_contours_d3(result, out_dir, kind_levels=kind_levels, progress_report=progress_report, clean_blobs=clean_blobs,
                                      blob_min_len=blob_min_len, blob_min_area_frac=blob_min_area_frac))
    rec("projections", export_projections_json(result, out_dir, progress_report=progress_report))
    rec("metrics", export_metrics_json(result, out_dir, progress_report=progress_report))
    rec("points3d", export_points3d_json(result, out_dir, progress_report=progress_report))
    rec("membership", export_membership_json(result, out_dir, progress_report=progress_report))
    if export_layout:
        rec("layout", export_layout_json(out_dir, progress_report=progress_report))
    if export_scales:
        rec("scales", export_scales_json(result, out_dir, progress_report=progress_report))

    # Flat summary
    all_paths = [p for group in manifest["written"].values() for p in group]
    manifest["summary"] = {
        "files": len(all_paths),
        "bytes": int(sum(os.path.getsize(p) for p in all_paths if os.path.exists(p))),
    }

    # Write manifest to disk
    mpath = _write_json(os.path.join(out_dir, "manifest.json"), manifest)
    _notify(progress_report, "write", kind="manifest", path=mpath)

    _notify(progress_report, "done", files=manifest["summary"]["files"], bytes=manifest["summary"]["bytes"])
    

