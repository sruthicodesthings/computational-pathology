"""Synthetic-only, reconstructed numerical examples; not original EPFL code."""
import numpy as np


def integrate_section_areas(ap_mm, areas_mm2):
    """Trapezoidal estimate of volume in mm^3 from sampled section areas."""
    z = np.asarray(ap_mm, dtype=float)
    a = np.asarray(areas_mm2, dtype=float)
    if z.ndim != 1 or a.ndim != 1 or len(z) != len(a) or len(z) < 2:
        raise ValueError("Provide at least two matching one-dimensional arrays")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("Positions/areas must be finite and areas nonnegative")
    order = np.argsort(z)
    z, a = z[order], a[order]
    if np.any(np.diff(z) <= 0):
        raise ValueError("AP coordinates must be unique")
    return float(np.sum((a[:-1] + a[1:]) * 0.5 * np.diff(z)))


def radius_limited_nearest(query_xyz_mm, reference_xyz_mm, radius_mm=0.05):
    """Return (index, distance_mm) or (None, distance_mm) when too distant."""
    q = np.asarray(query_xyz_mm, dtype=float)
    r = np.asarray(reference_xyz_mm, dtype=float)
    if q.shape != (3,) or r.ndim != 2 or r.shape[1] != 3 or len(r) == 0:
        raise ValueError("Expected query shape (3,) and nonempty references (N,3)")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)) or radius_mm <= 0:
        raise ValueError("Coordinates must be finite and radius positive")
    distances = np.linalg.norm(r - q, axis=1)
    i = int(np.argmin(distances))
    d = float(distances[i])
    return (i if d <= radius_mm else None), d


def apply_affine(points_xy, transform_3x3):
    """Apply a supplied 2D homogeneous affine transform (demonstration only)."""
    p = np.asarray(points_xy, dtype=float)
    t = np.asarray(transform_3x3, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or t.shape != (3, 3):
        raise ValueError("Expected (N,2) points and (3,3) matrix")
    h = np.column_stack([p, np.ones(len(p))])
    result = h @ t.T
    if np.any(np.isclose(result[:, 2], 0)):
        raise ValueError("Invalid homogeneous coordinate")
    return result[:, :2] / result[:, 2, None]
