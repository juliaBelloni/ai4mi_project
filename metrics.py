#!/usr/bin/env python3

# Dataset-independent segmentation evaluation metrics: IoU, Dice, HD95, ASD.

# Every function takes plain numpy label maps (not one-hot, not torch tensors)
# and a class value to score, so they work the same way for a 2-class (binary)
# problem or an N-class one, on any dataset.

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

def iou(pred: np.ndarray, gt: np.ndarray, c: int = 1) -> float:
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

    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64)
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64)

    # If class c is absent from both pred and gt (their union is empty)
    if union == 0:
        return 1.0

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

