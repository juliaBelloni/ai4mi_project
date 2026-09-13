# Metrics

[`metrics.py`](metrics.py) provides dataset-independent segmentation evaluation
metrics: 3D Dice, 95th-percentile Hausdorff Distance (HD95), and Average
Surface Distance (ASD). They operate on plain numpy label maps (not one-hot,
not torch tensors), take no dataset-specific assumptions (no hardcoded class
counts, no class-value scaling), and work for binary or multi-class volumes.

All four functions score a single class `c` at a time, except `dice`, which
loops over classes internally. For multi-class HD95/ASD, loop over class
values on the caller side (see examples below).

## `iou(pred, gt, c=1)`

Intersection-over-Union (Jaccard index) for one class. `pred`/`gt` are
integer (or boolean) label maps of identical shape; any dimensionality is
supported since IoU is purely voxel-counting.

```python
import numpy as np
from metrics import iou

pred = np.zeros((256, 256), dtype=np.uint8)
gt = np.zeros((256, 256), dtype=np.uint8)
pred[50:150, 50:150] = 1
gt[60:160, 60:160] = 1

score = iou(pred, gt, c=1)
```

**Edge case:** if class `c` is absent from both `pred` and `gt` (their union
is empty), `iou` returns `1.0` by convention â pred and gt trivially agree
there is nothing of that class, so this is treated as a perfect match rather
than an undefined `0/0`. This avoids NaNs silently poisoning any downstream
average over patients/classes.

## `dice(pred, gt, classes=None)`

Sorensen-Dice coefficient, derived algebraically from `iou` via
`Dice = 2*IoU / (1 + IoU)`, rather than reimplementing overlap counting.
Returns one score per class as a `np.ndarray`.

```python
from metrics import dice

# Score classes 1..4 of a 5-class (0=background) label volume:
scores = dice(pred_volume, gt_volume, classes=[1, 2, 3, 4])

# Or let it infer classes from the data:
scores = dice(pred_volume, gt_volume)
```

**Edge case:** a class absent from both `pred` and `gt` yields `Dice = 1.0`,
inherited directly from `iou`'s convention above (the algebra propagates it:
`IoU = 1.0` implies `Dice = 2*1/(1+1) = 1.0`).

## `hausdorff_distance_95(pred, gt, spacing, c=1)`

95th-percentile symmetric Hausdorff Distance, in mm, for one class of a 3D
volume. `spacing` is the physical voxel size `(sx, sy, sz)` in mm, in the
same axis order as `pred`/`gt` â e.g. the per-patient tuples saved to
`spacing.pkl` by [`slice_segthor.py`](slice_segthor.py).

```python
from metrics import hausdorff_distance_95

hd95 = hausdorff_distance_95(pred_volume, gt_volume, spacing=(0.98, 0.98, 2.5), c=1)

# Multi-class: loop over class values
hd95_per_class = [
    hausdorff_distance_95(pred_volume, gt_volume, spacing, c=k)
    for k in range(1, num_classes)
]
```

**Edge case:** a boundary distance is undefined when one side has no surface
to measure against.
- Class `c` absent from **both** `pred` and `gt`: returns `0.0` (no boundary
  disagreement is possible).
- Class `c` present in **exactly one** of `pred`/`gt`: returns `float("nan")`
  (there is no reference surface on the empty side).

## `average_surface_distance(pred, gt, spacing, c=1)`

Average Symmetric Surface Distance (ASD/ASSD), in mm: the mean of the pooled
pred-to-gt and gt-to-pred surface distances (one mean over every surface
voxel on both sides, not a mean of two per-direction means). Same input
convention as `hausdorff_distance_95`.

```python
from metrics import average_surface_distance

asd = average_surface_distance(pred_volume, gt_volume, spacing=(0.98, 0.98, 2.5), c=1)
```

**Edge case:** identical to `hausdorff_distance_95` â `0.0` when the class is
absent from both volumes, `float("nan")` when it's present in only one.

## Implementation notes

`hausdorff_distance_95` and `average_surface_distance` share the same
surface-extraction approach: binary erosion (`scipy.ndimage.binary_erosion`)
to find each mask's boundary voxels, then `scipy.ndimage.distance_transform_edt`
with `sampling=spacing` to get physically-scaled nearest-boundary distances in
each direction.
