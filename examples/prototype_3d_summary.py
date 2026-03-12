import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import mpase
from mpase.create_grid_planes import make_grid_from_bounds


def voxel_centers_from_edges(edges) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use true voxel centers so every occupied cell maps back to one XYZ coordinate.
    cx = 0.5 * (edges[0][:-1] + edges[0][1:])
    cy = 0.5 * (edges[1][:-1] + edges[1][1:])
    cz = 0.5 * (edges[2][:-1] + edges[2][1:])
    return cx, cy, cz


def occupancy_mask_3d(points3d: np.ndarray, edges) -> np.ndarray:
    # Count points per voxel on the shared grid, then reduce that to a simple occupancy mask.
    hist, _ = np.histogramdd(points3d, bins=edges)
    return hist > 0


def occupancy_points(mask: np.ndarray, edges) -> np.ndarray:
    # Convert occupied voxels back to XYZ coordinates so distance metrics stay interpretable.
    cx, cy, cz = voxel_centers_from_edges(edges)
    ix, iy, iz = np.where(mask)
    if len(ix) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack([cx[ix], cy[iy], cz[iz]]).astype(np.float32)


def iou_3d(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union + 1e-9)


def voxel_surface(mask: np.ndarray) -> np.ndarray:
    # Mark occupied voxels that touch the background in a 6-neighborhood.
    # This gives a cheap outer-surface approximation for a quick prototype.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return center & (~interior)


def surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, edges) -> Tuple[float, float]:
    surf_a = occupancy_points(voxel_surface(mask_a), edges)
    surf_b = occupancy_points(voxel_surface(mask_b), edges)
    if len(surf_a) == 0 or len(surf_b) == 0:
        return float("nan"), float("nan")

    tree_a = cKDTree(surf_a)
    tree_b = cKDTree(surf_b)
    d_ba, _ = tree_a.query(surf_b, k=1)
    d_ab, _ = tree_b.query(surf_a, k=1)

    mean_nn = float((d_ba.mean() + d_ab.mean()) / 2.0)
    hausdorff = float(max(d_ba.max(), d_ab.max()))
    return mean_nn, hausdorff


def summarize_3d(aligned_result: dict, grid_base: int = 48, pad_frac: float = 0.05) -> Dict[str, object]:
    # Build one shared voxel grid across all aligned sets so the 3D masks line up directly.
    aligned = aligned_result["aligned_points"]
    labels = aligned_result["labels"]
    edges, dims = make_grid_from_bounds(np.vstack(aligned), base=grid_base, pad_frac=pad_frac)

    masks = {}
    occupied_counts = {}
    surface_xyz = {}
    for lab, pts in zip(labels, aligned):
        mask = occupancy_mask_3d(pts, edges)
        masks[lab] = mask
        occupied_counts[lab] = int(mask.sum())
        surface_xyz[lab] = occupancy_points(voxel_surface(mask), edges)

    metrics = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_lab, b_lab = labels[i], labels[j]
            mean_nn, hausdorff = surface_distances(masks[a_lab], masks[b_lab], edges)
            metrics.append(
                {
                    "A": a_lab,
                    "B": b_lab,
                    "voxel_iou": iou_3d(masks[a_lab], masks[b_lab]),
                    "surface_meanNN": mean_nn,
                    "surface_hausdorff": hausdorff,
                }
            )

    return {
        "labels": labels,
        "grid_dims": [int(x) for x in dims],
        "occupied_voxels": occupied_counts,
        "metrics": metrics,
        "masks": masks,
        "surface_xyz": surface_xyz,
        "edges": edges,
    }


def _style_3d_axes(ax, title: str):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))


def plot_aligned_points(aligned_result: dict, out_dir: str) -> str:
    # Show the aligned point clouds directly so it is easy to sanity-check registration.
    labels = aligned_result["labels"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx, (lab, pts) in enumerate(zip(labels, aligned_result["aligned_points"])):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=6,
            alpha=0.35,
            color=colors[idx % len(colors)],
            label=lab,
        )

    _style_3d_axes(ax, "Aligned 3D points")
    ax.legend(frameon=False, loc="upper right")
    path = os.path.join(out_dir, "aligned_points_3d.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_voxel_surfaces(summary: dict, out_dir: str) -> str:
    # Plot occupied surface voxels instead of full volumes so the structure stays visible.
    labels = summary["labels"]
    surface_xyz = summary["surface_xyz"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for idx, lab in enumerate(labels):
        surf = surface_xyz[lab]
        color = colors[idx % len(colors)]
        ax1.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=12, alpha=0.75, color=color, label=lab)

    # Put both labels in separate panels too, because overlap can hide thin differences.
    if len(labels) >= 1:
        surf = surface_xyz[labels[0]]
        ax2.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=12, alpha=0.75, color=colors[0], label=labels[0])
    if len(labels) >= 2:
        surf = surface_xyz[labels[1]]
        ax2.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=12, alpha=0.30, color=colors[1], label=labels[1])

    _style_3d_axes(ax1, "Voxel surface comparison")
    _style_3d_axes(ax2, "Voxel surface overlay")
    ax1.legend(frameon=False, loc="upper right")
    ax2.legend(frameon=False, loc="upper right")

    path = os.path.join(out_dir, "voxel_surface_3d.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_json(summary: dict, out_dir: str) -> str:
    # Strip large arrays before writing JSON so the summary stays small and readable.
    serializable = {
        "labels": summary["labels"],
        "grid_dims": summary["grid_dims"],
        "occupied_voxels": summary["occupied_voxels"],
        "metrics": summary["metrics"],
    }
    path = os.path.join(out_dir, "summary_3d.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path


def main():
    # Build two related synthetic 3D point sets so the output is easy to inspect visually.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(800, 3)).astype(np.float32)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    B = (A @ rot.T) + 0.06 * rng.normal(size=A.shape).astype(np.float32)

    aligned = mpase.align_points(
        points_list=[A, B],
        labels=["A", "B"],
        align_mode="auto",
    )
    summary = summarize_3d(aligned, grid_base=48, pad_frac=0.05)

    out_dir = os.path.join("examples", "prototype_3d_out")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = save_summary_json(summary, out_dir)
    aligned_fig = plot_aligned_points(aligned, out_dir)
    surface_fig = plot_voxel_surfaces(summary, out_dir)

    print("3D summary prototype result")
    print(
        json.dumps(
            {
                "labels": summary["labels"],
                "grid_dims": summary["grid_dims"],
                "occupied_voxels": summary["occupied_voxels"],
                "metrics": summary["metrics"],
            },
            indent=2,
        )
    )
    print(f"saved summary: {summary_path}")
    print(f"saved figure: {aligned_fig}")
    print(f"saved figure: {surface_fig}")


if __name__ == "__main__":
    main()
