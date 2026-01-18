import numpy as np
# ---------------------------- Grid / Planes ----------------------------
PLANE_FROM_AXIS = {'z': 'XY', 'x': 'YZ', 'y': 'XZ'}
AXPAIR = {'z': (0,1), 'x': (1,2), 'y': (0,2)}

def make_grid_from_bounds(P: np.ndarray, base=160, pad_frac=0.05):
    # find min and max along each axis (x, y, z)
    # This gives the bounding box of all points P
    mn, mx = P.min(0), P.max(0)
    # Compute the extent = size of the box in each dimension
    ext = mx - mn
    # add padding, expand both min and max by this pad amount
    pad = pad_frac * float(ext.max()); mn -= pad; mx += pad; ext = mx - mn
    # decide how finely to scale the grid
    # base = target size of the longest axis (e.g. 160)
    scale = base / float(ext.max() + 1e-8)
    # compute the number of bins (voxels) along each axis
    # clamp to [32, 512] to avoid too coarse or too fine grids
    dims = np.clip(np.ceil(ext * scale).astype(int), 32, 512)
    # compute the bin edges along each axis
    edges = [np.linspace(mn[i], mx[i], dims[i] + 1, dtype=np.float32) for i in range(3)]
    return edges, dims

# def grid_centers_from_edges(ex, ey):
#     # Number of pixels along X and Y for this plane.
#     # Convert bin edges into number of bins (pixels). If edges have length N+1, there are N bins.
#     # The raster mask needs exact dimensions
#     nx = len(ex) - 1; ny = len(ey) - 1
#     # Create arrays of pixel center coordinates along X and Y.
#     # When rasterizing points, we need to know where pixel centers are to place each point in the right bin.
#     xs = np.linspace(ex[0], ex[-1], nx)
#     ys = np.linspace(ey[0], ey[-1], ny)
#     return xs, ys

def grid_centers_from_edges(ex, ey):
    """
    Return true bin centers given edges ex (len nx+1) and ey (len ny+1).
    Centers are midpoints between consecutive edges.
    """
    xs = 0.5 * (ex[:-1] + ex[1:])
    ys = 0.5 * (ey[:-1] + ey[1:])
    return xs, ys


# Projects a 3D point set P onto a 2D plane by dropping one axis.
def project_plane(P: np.ndarray, axis: str) -> np.ndarray:
    # The dictionary maps the dropped axis ('x', 'y', or 'z') to the two axes to keep.
    d = {'x': [1,2], 'y': [0,2], 'z': [0,1]}[axis]
    # P[:, d] slices columns to keep just those two coordinates.
    return P[:, d]

def rasterize_points(points2d: np.ndarray, xs, ys, disk_px=2, ex=None, ey=None) -> np.ndarray:
    """Rasterize points to a binary mask with small disks (radius=disk_px pixels).
    Takes 2D points and turns them into a pixel grid (mask).
    """
    # get the height (ny) and width (nx) of the image grid
    ny, nx = len(ys), len(xs)
    # initialize a binary image of size (ny, nx) with all set to False
    img = np.zeros((ny, nx), dtype=bool)
    # create an array of integer offset values from -disk_px to +disk_px
    rr = np.arange(-disk_px, disk_px+1)
    # build 2D grids of x and y offsets covering the square
    XX, YY = np.meshgrid(rr, rr, indexing='xy')
    # create a boolean mask of the circle (disk)
    # for each square check if x^2 + y^2 is within the radius squared
    # True for pixels inside the disk, False outside
    disk = (XX**2 + YY**2) <= (disk_px**2)
    # half-height and half-width of the disk
    dh, dw = disk.shape[0]//2, disk.shape[1]//2
    
    if ex is not None and ey is not None:
        # --- FIX 3: bin by EDGES (right-exclusive except last bin), matches HDR binning ---
        x_idx = np.clip(np.searchsorted(ex, points2d[:,0], side="right") - 1, 0, nx-1)
        y_idx = np.clip(np.searchsorted(ey, points2d[:,1], side="right") - 1, 0, ny-1)
    else:
        # convert each point's (x,y) coordinates to pixel indices
        # np.searchsorted finds the index where each point would fit in the sorted xs/ys
        # subtract 1 to get the pixel to the left/below the point
        # clip to ensure indices are within image bounds [0, nx-1] or [0, ny-1]
        x_idx = np.clip(np.searchsorted(xs, points2d[:,0]) - 1, 0, nx-1)
        y_idx = np.clip(np.searchsorted(ys, points2d[:,1]) - 1, 0, ny-1)
    
    # loop over each point's pixel index
    for y, x in zip(y_idx, x_idx):
        # compute the bounding box of the disk, clipped to image boundaries
        # Makes sure the disk doesn’t go outside the image array.
        y0, y1 = max(0, y-dh), min(ny, y+dh+1)
        x0, x1 = max(0, x-dw), min(nx, x+dw+1)
        # adjust the starting index inside the disk mask when part of the disk if clipped
        # If the disk goes off the edge, we can’t take the full mask
        # we need to slice it starting further in.
        dy0 = 0 if y0==y-dh else (y-dh - y0)
        dx0 = 0 if x0==x-dw else (x-dw - x0)
        # Overlay the disk on the image.
        # |= means logical OR — so if any disk covers a pixel, that pixel becomes True.
        # The slices [dy0:…] and [dx0:…] ensure we only take the part of the disk mask that fits inside the image.
        # This step “paints” the disk onto the raster mask.
        img[y0:y1, x0:x1] |= disk[dy0:dy0+(y1-y0), dx0:dx0+(x1-x0)]
    return img


def points_to_pixel_indices(points2d: np.ndarray, xs, ys):
    """
    Map each 2D point to integer pixel indices (x_idx, y_idx)
    on the same grid used by rasterize_points.

    points2d: [N, 2] in the same coordinate system as xs/ys.
    xs, ys: 1D arrays of grid center coordinates.
    """
    ny, nx = len(ys), len(xs)
    x_idx = np.clip(np.searchsorted(xs, points2d[:, 0]) - 1, 0, nx - 1)
    y_idx = np.clip(np.searchsorted(ys, points2d[:, 1]) - 1, 0, ny - 1)
    return x_idx, y_idx

