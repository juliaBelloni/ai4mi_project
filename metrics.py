#!/usr/bin/env python3

# Dataset-independent segmentation evaluation metrics: IoU, Dice, HD95, ASD.

# Every function takes plain numpy label maps (not one-hot, not torch tensors)
# and a class value to score, so they work the same way for a 2-class (binary)
# problem or an N-class one, on any dataset.

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

def iou(pred: np.ndarray, gt: np.ndarray, c: int = 1, eta: float = 1e-8) -> float:
    """
    Parameters
    ----------
    pred, gt:
        Integer (or boolean) label maps of identical shape. Any
        dimensionality is supported (2D slice or 3D volume)
    c:
        The class value to score. For an already-boolean mask, the default
        c=1 selects the foreground.
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)

    pred_mask = pred == c
    gt_mask = gt == c

    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64) + eta
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64) + eta

    return float(intersection / union)

def dice(pred: np.ndarray, gt: np.ndarray, classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Parameters
    ----------
    pred, gt:
        Integer label maps of identical shape.
    classes:
        The class values to score.

    Returns
    -------
    np.ndarray
        1D array of shape (len(classes),), one Dice score per class in
        [0, 1]. A class absent from both pred and gt yields Dice = 1.0,
    """
    if classes is None:
        classes = sorted(set(np.unique(pred).tolist()) | set(np.unique(gt).tolist()))

    ious = np.array([iou(pred, gt, c) for c in classes], dtype=np.float64)

    return 2 * ious / (1 + ious)

def _mask_border(mask: np.ndarray) -> np.ndarray:
    """Boolean array marking the surface (boundary) voxels of a binary mask."""
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def _surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray,
                        spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns `(pred_to_gt, gt_to_pred)`: the distance from every surface voxel of `pred_mask` to the nearest surface voxel of `gt_mask`, and vice versa.
    """
    # callers ensure masks are non-empty
    assert pred_mask is not None and gt_mask is not None
    pred_border = _mask_border(pred_mask)
    gt_border = _mask_border(gt_mask)

    dt_gt = ndimage.distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)

    return dt_gt[pred_border], dt_pred[gt_border]


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], c: int = 1) -> float:
    """

    Parameters
    ----------
    pred, gt:
        3D integer (or boolean) label maps of identical shape, `(X, Y, Z)`.
    spacing:
        Physical voxel size (sx, sy, sz) in mm, matching the axis order of
        pred/gt.
    c:
        The class value to score.

    Returns
    -------
    float
        HD95 in mm: `max(P95(pred->gt distances), P95(gt->pred distances))`.
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.ndim == len(spacing), (pred.shape, spacing)

    pred_mask = pred == c
    gt_mask = gt == c

    # no boundary disagreement is possible
    if not pred_mask.any() and not gt_mask.any():
        return 0.0

    #  no reference surface on the empty side
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")

    pred_to_gt, gt_to_pred = _surface_distances(pred_mask, gt_mask, spacing)

    return float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))

def average_surface_distance(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], c: int = 1) -> float:
    """

    Parameters
    ----------
    pred, gt:
        3D label maps of identical shape (X, Y, Z).
    spacing:
        Physical voxel size (sx, sy, sz) in mm, matching the axis order of
        pred/gt
    c:
        The class value to score.

    Returns
    -------
    float
        The mean, in mm, of the pooled pred->gt and gt->pred surface
        distances (one mean over every surface voxel on both sides).
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.ndim == len(spacing), (pred.shape, spacing)

    pred_mask = pred == c
    gt_mask = gt == c

    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")

    pred_to_gt, gt_to_pred = _surface_distances(pred_mask, gt_mask, spacing)

    return float(np.mean(np.concatenate([pred_to_gt, gt_to_pred])))

