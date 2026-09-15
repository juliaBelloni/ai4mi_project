# Segmentation post-processing

Post-processing removes selected connected regions from predicted segmentation
masks before evaluation. The functions operate on full 3D NumPy label maps,
not individual slices or probability maps. Label `0` is background.

[`eval.py`](eval.py) loads compressed NIfTI files directly from a folder such as
`volumes/segthor/ce`, with one `<patient_id>.nii.gz` per patient. Filtering runs
in memory after loading and before computing metrics. Each filter returns a
copy with the original shape, dtype, and surviving class labels; discarded
voxels become background. Ground truth and source files are not overwritten.
By default, only metric CSVs are saved; `eval.py --save` also saves the evaluated
prediction volumes after optional post-processing.

### Spatial alignment assumption

Evaluation compares predictions and ground truth at matching array indices.
It checks that their shapes match, but does not resample or align images using
their NIfTI affine matrices. Predictions must therefore already use the same
voxel ordering and grid as ground truth. Distance metrics use ground-truth
voxel spacing.

In the five prediction files used for the initial comparison, the shapes match
ground truth, but the predictions have identity affines and 1 mm spacing while
ground truth has different spatial metadata. The Dice comparison assumes those
arrays are aligned despite this metadata discrepancy; matching shapes alone
do not verify alignment. The post-processing functions do not repair metadata
or establish spatial correspondence.

## Methods

### No filtering (baseline)

Evaluate the original prediction labels. This is the default in `eval.py`
(`--postprocessing none`) and the `baseline` entry in the comparison script.
It has no filtering hyperparameters.

### Top-k connected components per class

Implemented as `keep_largest_connected_components` in
[`postprocessing.py`](postprocessing.py).

For each selected foreground class, create a binary mask and label its 3D
connected components using SciPy's `ndimage.label`. Rank components by voxel
count, keep the largest k, and replace all other components with background.
Different classes never connect to each other for this operation.

For example, if one class has components of 500, 100, and 10 voxels, `k=2`
keeps the 500- and 100-voxel regions. If a class has fewer than k components,
all its components survive. Equal-size ties favor the component encountered
first in array traversal order. Absent classes have no effect.

| Python parameter | CLI option in `eval.py` | Default | Meaning |
|---|---|---|---|
| `k` | `--top_k` | `1` | Positive integer; maximum number of components retained **for each selected class**. The same k applies to all selected classes. |
| `classes` | `--postprocessing_classes` | `None` | Process all nonzero prediction labels by default. Supply labels such as `1 2` to filter only those classes. Unselected classes are unchanged; background is always skipped. |
| `connectivity` | `--connectivity` | `26` | Which neighboring voxels count as connected: 6, 18, or 26 (see below). |

Enable this method with `--postprocessing largest_connected_components`.
This ranks regions by relative size; it does not apply a minimum-size threshold
or determine anatomical plausibility. A valid disconnected organ fragment can
be removed even if it is large.

### Top-k connected components across combined foreground

Implemented experimentally as `keep_foreground_components(volume, k)` in
[`postprocessing.py`](postprocessing.py).

Create a single binary mask from `volume != 0`, label its connected components,
and keep the largest k components across that entire mask. Restore the original
class labels for surviving voxels. Touching regions of different classes count
as one component, but their labels are not merged in the returned segmentation.

| Parameter | Setting | Meaning |
|---|---|---|
| `k` | Required; comparison uses `1`, `2`, `3` | Maximum number of components across **all foreground classes together**. |
| Connectivity | Fixed at `26` in this helper | Includes face, edge, and corner neighbors. |
| Class selection | All nonzero labels | No class-selection argument is exposed. |

A disconnected correctly predicted organ may be removed. Conversely, a
false-positive region touching another organ may survive as part of a larger
component. This method is available in the comparison script and through the
Python callable interface, but is not an `eval.py` CLI choice.

### Remove small connected components per class

Implemented experimentally as `remove_small_components(volume, min_size)` in
[`postprocessing.py`](postprocessing.py).

Label components independently for every foreground class and remove only
components with fewer than `min_size` voxels. Any number of components can
survive, provided each meets the threshold. A component exactly at the threshold
is retained; a class can disappear entirely if all its components are too small.

For components of 500, 100, and 10 voxels, `min_size=100` keeps the first two.

| Parameter | Setting | Meaning |
|---|---|---|
| `min_size` | Required; comparison uses `10`, `100`, `1000` | Minimum retained component size in **voxels**, not mm³. Use a positive integer. |
| Connectivity | Fixed at `26` in this helper | Includes face, edge, and corner neighbors. |
| Class selection | All nonzero labels, independently | No class-selection argument is exposed. |

Because the threshold is a voxel count, its physical volume depends on scan
spacing. This method can remove small valid structures as well as false positives.
It is available in the comparison script and through the Python callable
interface, but is not an `eval.py` CLI choice.

## Connectivity

| Value | Neighbor relationship in 3D |
|---|---|
| `6` | Voxels sharing a face. |
| `18` | Voxels sharing a face or edge. |
| `26` | Voxels sharing a face, edge, or corner. |

Larger neighborhoods can join regions that would be separate under a smaller
neighborhood. Connectivity is defined on the voxel grid, not by a physical
distance in millimeters. All variants in the current comparison use `26`.

## Evaluate a method

Run from the project root, adjusting `--gt_pattern` to your ground-truth files.
Use distinct output paths to retain results from different settings:

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --dest results/segthor/ce/baseline.csv

python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing largest_connected_components --top_k 2 --connectivity 26 \
    --dest results/segthor/ce/components_k2.csv
```

Add `--postprocessing_classes 1 2` to filter only those labels. This controls
filtering, not which classes are scored. By default evaluation computes Dice,
HD95, and average surface distance; use `--metrics dice` to evaluate Dice only.
Each run writes per-patient/per-class scores and a corresponding `_summary.csv`.
See [`metrics.md`](metrics.md) for metric definitions and other evaluation options.

### Save the processed volumes

Add `--save` to write one `<patient_id>.nii.gz` per patient. By default, a CSV
destination of `results/segthor/ce/components_k2.csv` produces volumes in
`results/segthor/ce/components_k2_volumes/`. Set `--save_folder` to choose another
folder; this option requires `--save`:

```sh
python eval.py --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --postprocessing largest_connected_components --top_k 2 \
    --dest results/segthor/ce/components_k2.csv \
    --save --save_folder volumes/segthor/ce_k2
```

The saved labels are the predictions used to compute the metrics. Saving
preserves the prediction's affine, qform/sform codes, voxel spacing, and header
metadata; it does not copy ground-truth geometry or repair existing metadata
discrepancies. Output paths matching the patient's prediction or ground truth
are rejected. Existing files in the output folder are replaced, so use a
different folder for each configuration you want to keep. With
`--postprocessing none`, `--save` saves the unfiltered predictions.

Python callers can pass `save_folder=Path(...)` to `evaluate_patient` or
`evaluate_dataset`. The comparison script continues to save CSVs only.

Python callers can pass any callable taking and returning a 3D label map to
`evaluate_patient` or `evaluate_dataset`:

```python
from functools import partial
from pathlib import Path
from eval import evaluate_dataset
from postprocessing import keep_largest_connected_components

rows = evaluate_dataset(
    Path('volumes/segthor/ce'),
    'data/segthor_part1/train/{id_}/GT.nii.gz',
    metrics=['dice'],
    postprocess=partial(keep_largest_connected_components, k=2, connectivity=26),
)
```

For the experimental methods, import their helpers from `postprocessing`
and use `partial(keep_foreground_components, k=2)` or
`partial(remove_small_components, min_size=100)` as the callable.

## Reproduce the comparison

```sh
python compare_postprocessing.py \
    --pred_folder volumes/segthor/ce \
    --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
    --dest_folder results/postprocessing_comparison
```

The script evaluates the baseline, both top-k methods at k=1, 2, and 3, and
per-class minimum sizes of 10, 100, and 1,000 voxels. These settings are defined
in the script's `methods` construction, not exposed as CLI flags. Its three CLI
options select the prediction folder, ground-truth pattern, and output folder;
the defaults match the command above.

It computes **Dice only** and writes `per_patient.csv` and `summary.csv`.
The summary includes mean Dice per class, the mean over all patient/class pairs,
the mean change versus baseline, and counts of improved, worsened, or unchanged
patient/class pairs (changes within `1e-12` count as unchanged).

In the initial comparison on five volumes and foreground labels 1, 2, and 3,
no tested filter improved overall mean Dice over the baseline (`0.589462`).
Removing components smaller than 10 voxels was nearly neutral (`0.589447`).
Combined-foreground top-k outperformed per-class top-k at each tested k, but
still scored below baseline. These findings concern those predictions and
settings; HD95 and average surface distance were not compared.
