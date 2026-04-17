"""
Synthetic point cloud generators for MPASE evaluation.

Shapes:
  - s_shape  : S-curve in 3D
  - helix    : helical coil
  - blob     : Gaussian blob
  - two_blob : two separated Gaussian blobs

Each function returns an (N, 3) float32 numpy array.
"""

import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic")


def s_shape(n: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, n)
    x = np.sin(t)
    y = np.sign(np.sin(t)) * (1 - np.cos(t))
    z = rng.normal(0, 0.05, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def helix(n: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, n)
    x = np.cos(t) + rng.normal(0, 0.05, n)
    y = np.sin(t) + rng.normal(0, 0.05, n)
    z = t / (4 * np.pi) + rng.normal(0, 0.02, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def blob(n: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (n, 3))
    return pts.astype(np.float32)


def two_blob(n: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    half = n // 2
    a = rng.normal([-2, 0, 0], 0.8, (half, 3))
    b = rng.normal([2, 0, 0], 0.8, (n - half, 3))
    return np.vstack([a, b]).astype(np.float32)


SHAPES = {
    "s_shape": s_shape,
    "helix": helix,
    "blob": blob,
    "two_blob": two_blob,
}


def apply_rigid(pts: np.ndarray, R: np.ndarray, t: np.ndarray,
                noise_frac: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Apply rotation R and translation t, then add optional isotropic Gaussian noise.
    noise_frac is relative to the point cloud extent (max range across all axes).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = pts @ R.T + t
    if noise_frac > 0:
        extent = np.ptp(pts, axis=0).max()
        out = out + rng.normal(0, noise_frac * extent, out.shape)
    return out.astype(np.float32)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random 3D rotation via QR decomposition of a random matrix."""
    M = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def random_translation(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    return (rng.uniform(-scale, scale, 3)).astype(np.float32)


def save_all(n: int = 1000, seed: int = 0):
    """Save all shapes as .npy files to data/synthetic/."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name, fn in SHAPES.items():
        pts = fn(n=n, rng=rng)
        path = os.path.join(OUT_DIR, f"{name}.npy")
        np.save(path, pts)
        print(f"Saved {path}  shape={pts.shape}")


if __name__ == "__main__":
    save_all()
