"""Reference-anchored coordinate translation; NOT a full ABBA transform reconstruction."""
from __future__ import annotations
import numpy as np

def root_anchored_translation(points: np.ndarray, exported_root: np.ndarray,
                              expected_root: np.ndarray) -> np.ndarray:
    """Translate 3D points so an exported reference anchor maps to its expected location.

    This cannot correct rotation, scale, deformation, or incorrect AP ordering.
    """
    pts = np.asarray(points, dtype=float)
    source = np.asarray(exported_root, dtype=float)
    target = np.asarray(expected_root, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or source.shape != (3,) or target.shape != (3,):
        raise ValueError("Expected points (N,3) and two (3,) reference coordinates")
    if not (np.isfinite(pts).all() and np.isfinite(source).all() and np.isfinite(target).all()):
        raise ValueError("Non-finite coordinates")
    return pts + (target-source)

def atlas_bounds_mask(points: np.ndarray, bounds_mm: np.ndarray) -> np.ndarray:
    """Check coordinates against caller-supplied [min,max] for each of three axes."""
    pts = np.asarray(points, dtype=float)
    bounds = np.asarray(bounds_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or bounds.shape != (3,2):
        raise ValueError("Expected points (N,3), bounds (3,2)")
    if not np.isfinite(bounds).all() or np.any(bounds[:,0] >= bounds[:,1]):
        raise ValueError("Invalid bounds")
    return np.isfinite(pts).all(axis=1) & np.all((pts >= bounds[:,0]) & (pts <= bounds[:,1]), axis=1)
