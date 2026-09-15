"""Post-processing functions for 3D numpy segmentation label maps."""

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Expected a 3D label map.")
    if not (np.issubdtype(volume.dtype, np.integer) or volume.dtype == np.bool_):
        if not (np.issubdtype(volume.dtype, np.floating)
                and np.isfinite(volume).all()
                and np.equal(volume, np.floor(volume)).all()):
            raise ValueError("Expected finite integer-valued labels.")
    return volume


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def keep_largest_connected_components(
    volume: np.ndarray,
    k: int = 1,
    classes: Optional[Sequence[int]] = None,
    connectivity: int = 26,
) -> np.ndarray:
    """Keep up to k largest 3D components independently for each selected class.

    Components are ranked by voxel count. Connectivity is 6 (faces), 18
    (faces and edges), or 26 (faces, edges and corners). Equal-size ties
    favor the component encountered first in array traversal order.

    By default, process all nonzero labels. Label 0 is background and is
    never filtered. Unselected classes are unchanged; discarded voxels
    become background. Returns a copy with the original shape and dtype.
    This is size-based filtering, not a test of anatomical plausibility.
    """
    volume = _validate_volume(volume)
    _validate_positive_integer(k, "k")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")

    selected = np.unique(volume) if classes is None else classes
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    result = volume.copy()
    for c in selected:
        if c == 0:
            continue
        mask = volume == c
        components, count = ndimage.label(mask, structure=structure)
        if count <= k:
            continue
        sizes = np.bincount(components.ravel())[1:]
        largest = np.argsort(-sizes, kind="stable")[:k] + 1
        keep = np.zeros(count + 1, dtype=bool)
        keep[largest] = True
        result[mask & ~keep[components]] = 0
    return result


def keep_foreground_components(volume: np.ndarray, k: int) -> np.ndarray:
    """Connect all foreground labels together, but preserve surviving labels."""
    volume = _validate_volume(volume)
    keep = keep_largest_connected_components(volume != 0, k=k, connectivity=26)
    result = volume.copy()
    result[~keep] = 0
    return result


def remove_small_components(volume: np.ndarray, min_size: int) -> np.ndarray:
    """Remove per-class components with fewer than min_size voxels."""
    volume = _validate_volume(volume)
    _validate_positive_integer(min_size, "min_size")
    result = volume.copy()
    structure = ndimage.generate_binary_structure(3, 3)
    for c in np.unique(volume):
        if c == 0:
            continue
        mask = volume == c
        components, _ = ndimage.label(mask, structure)
        sizes = np.bincount(components.ravel())
        keep = sizes >= min_size
        keep[0] = False
        result[mask & ~keep[components]] = 0
    return result
