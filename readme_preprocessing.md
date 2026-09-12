# Testing
## Added flag: `--test_pipeline`

The flag uses the first two sorted patients (Patient_01 and Patient_02), creates smoke data if its directory is missing, and runs one training/validation epoch. Existing smoke directories are reused as-is.

# Preprocessing -> USE A NEW --DATA_DIR so you actually change the data
## Added flags: `--hu_min` and `--hu_max` -> for this dataset use -1000, 300! 

Supply both flags to enable HU windowing. Without them, the original per-volume
Min-Max normalization stays unchanged. Bounds must be finite and `hu_min < hu_max`.

TorchIO `Clamp` clips CT intensities to the supplied bounds, then `RescaleIntensity`
uses those fixed bounds (not each scan's observed extrema) to map HU to `[0, 1]`:

```text
normalized = (clip(HU, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
```

The current pipeline still stores 8-bit PNG images: normalized values are multiplied
by 255 and quantized, then divided by 255 by the training loader. Masks are unchanged.

For example, test the `[-200, 300]` candidate window in one command:

```bash
python main.py --test_pipeline --hu_min -200 --hu_max 300 \
  --data_dir data/SEGTHOR_smoke_hu_m200_300 \
  --dest results/segthor/smoke_hu_m200_300
```

Use a fresh `--data_dir` when changing bounds: the simple smoke cache checks only
whether the directory exists. The same HU flags also work directly in `slice_segthor.py`.
The notebook compares candidate windows; these example bounds are not automatic defaults.

## Added flag: `--target_spacing`

In-plane pixel spacing (`dx`/`dy` from the CT header) varies per patient (roughly 0.9-1.4mm),
but the baseline always resizes every patient's slice to the same fixed `--shape` pixel grid
regardless of that native spacing — so the same pixel count covers a different physical area
for different patients.

Supply `--target_spacing` (mm per in-plane pixel) to fix this: each patient's slice is first
resized so 1 pixel consistently represents that physical spacing, then center-cropped (or
zero/background-padded, if smaller) to the fixed `--shape` so batching still works. Without
the flag, behavior is unchanged from the baseline (a direct resize straight to `--shape`,
ignoring native spacing).

```bash
python slice_segthor.py --source_dir data/segthor_part1 --dest_dir data/SEGTHOR_spacing1 \
  --shape 256 256 --target_spacing 1.0
```

Use a fresh `--data_dir`/`--dest_dir` when changing this, same as HU windowing above — it
changes the stored pixel content, not just how it's loaded. Also works with `--test_pipeline`
via `python main.py --test_pipeline --target_spacing 1.0 --data_dir data/SEGTHOR_smoke_spacing1`.

**Recommended value:** use the median in-plane spacing across the training patients rather
than an arbitrary number — this is the standard approach (e.g. nnU-Net resamples to the
dataset's median spacing by default). Per `preprocessing_params.ipynb`, that's currently
**0.9766mm** for this dataset. No automatic detection yet; pass it explicitly.

# Data correction

## Added flag: `--fix_aorta_esophagus`

`GT.nii.gz` merges the aorta into the esophagus label (`1`) instead of using its own
label (`4`), for every patient checked. Confirmed three ways: raw label values in
`GT.nii.gz` never include `4`; a leftover corrected annotation for `Patient_07`
(`GT2.nii.gz`) shows the same voxels correctly split as `1`/`4`; and the original
SegTHOR challenge lists aorta as one of its 4 target organs (`readme.md`), so this
is a data bug in how this course's copy was packaged, not an intentional 4-class
dataset. Reported to the course; fix it before training regardless of the cause.

Splits them apart using shape alone, no CT intensity needed: the aorta is much
thicker than the esophagus, so eroding the merged region by a few voxels leaves
only the aorta's core; dilating that core back (clipped to the original merged
region) recovers its full extent. Small leftover islands on either side are
reassigned to the other class, since both the real esophagus and the real aorta
are each a single connected tube. See `split_merged_aorta_esophagus()` in
`slice_segthor.py` for the exact steps.

Validated against `Patient_07`'s known-correct split (`GT2.nii.gz`, the only
patient with a ground-truth answer): **0.99 aorta Dice, 0.97 esophagus Dice**.
Also tried and rejected as worse or unnecessarily complex: pure HU-intensity
thresholding (aorta and esophagus overlap too much in raw intensity to separate
this way), watershed and random-walker variants (both plateaued below plain
erosion/dilation even after tuning), and a per-voxel geometry classifier trained
against heart/trachea position (matched this method's accuracy but added a
training dependency and per-patient classifier for no accuracy gain once the
same island cleanup was applied to the simpler method).

```bash
python slice_segthor.py --source_dir data/segthor_part1 --dest_dir data/SEGTHOR_fixed \
  --shape 256 256 --fix_aorta_esophagus
```

Use a fresh `--data_dir`/`--dest_dir` when enabling this, same as HU windowing and
`--target_spacing` above — it changes the stored pixel content, not just how it's
loaded. Also works with `--test_pipeline` via
`python main.py --test_pipeline --fix_aorta_esophagus --data_dir data/SEGTHOR_smoke_fixed`.

Known limitation: one patient (`Patient_03`) is a genuine outlier where this
shape-only method underperforms relative to the others checked — worth a manual
look before trusting it blindly on every patient.

# Training-time slice filtering

## Added flag: `--drop_empty_slices`

Many axial slices (near the top/bottom of the chest CT stack) contain no organ of any
class — pure background. Training on the natural, highly imbalanced distribution wastes
gradient updates on trivially-easy all-background slices.

`--drop_empty_slices FRACTION` (0-1) randomly drops that fraction of purely-empty-GT slices
from the **training** set only, using a fixed seed for reproducibility. **Validation is never
filtered** — it must keep every slice, including empty ones, so the reported metric still
reflects real-world performance. Default `0` keeps every slice, matching the baseline exactly.

```bash
python main.py --dataset SEGTHOR --mode full --epochs 25 \
    --dest results/segthor/ce_dropempty --gpu --drop_empty_slices 0.9
```

Note: this filters slices with **no organ of any class** present. It's unrelated to, and does
not fix, individual classes (e.g. aorta) being absent from a slice that still has other organs
present — that's a separate metric-reporting concern, not a training-data-balance one.


# Augmentation

## Added flag: `--augment`

The flag turns on **online augmentation** of training slices in
[`SliceDataset`](dataset.py). Its purpose is to help the model tolerate small
positioning differences and image noise. Whether it improves segmentation must be tested still.

[`main.py`](main.py) defines the flag with `action='store_true'`, so augmentation
is disabled by default. It passes the flag only to the training dataset, and
`SliceDataset` also requires `subset == 'train'`. Validation and test data are
never augmented. 

### What it does

| Operation | Probability per loaded training slice | Parameters | Applied to |
| --- | --- | --- | --- |
| Small in-plane rotation | 50% | Angle sampled uniformly from -5 to +5 degrees | CT image and mask, using the same angle |
| Additive Gaussian noise | 25% | Zero-mean noise with standard deviation sampled uniformly from 0 to 0.1 | CT image only |

The two decisions are independent: a slice can receive either operation, both,
or neither. Random values are sampled again whenever a slice is loaded. No
augmented images are written to disk, and the number of samples per epoch stays
the same. Offline augmentation would instead save transformed copies in advance.

Rotation happens before the original image and mask transforms. CT images use
bilinear interpolation; masks use nearest-neighbor interpolation to preserve
their encoded class values (`0, 63, 126, 189, 252` for SegTHOR). The output keeps
its original dimensions. Newly exposed pixels are filled with zero: black for
the CT image and background for the mask. Outer corners can be clipped.

Noise is added after `img_transform` converts the CT image into a floating-point
tensor in `[0, 1]`. Values are clamped back to `[0, 1]` afterward. Noise never
changes the mask. `gt_transform` still converts the mask to one-hot labels.

**Noise strength:** the current maximum standard deviation is `0.1`. The mild
starting setting discussed for this project is `0.01`, ten times smaller. To use
that setting, change the sampling line in `dataset.py` to:

```python
noise_std = torch.empty(()).uniform_(0.0, 0.01).item()
```

### Why these choices fit this CT pipeline

- **No flips:** preserve the anatomical asymmetry.
- **Small rotations:** vary positioning without introducing large orientation changes. (since the person is always scanned in the same way)
- **Gaussian noise:** a practical robustness augmentation for reconstructed CT
  images, also used in [CT segmentation research](https://www.nature.com/articles/s41598-020-67544-y). See
  [CT noise-model study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5783547/).
- **Full slices:** there is no random patch sampling, scaling, translation,
  gamma adjustment, or elastic deformation in this implementation.

### Commands

After correcting the image-loading lines, enable augmentation with:

```bash
python main.py --dataset SEGTHOR --mode full --epochs 25 \
    --dest results/segthor/ce_aug --gpu --augment
```

not using the flag triggers the standard no augmentation.

```bash
python main.py --dataset SEGTHOR --mode full --epochs 25 \
    --dest results/segthor/ce --gpu
```
