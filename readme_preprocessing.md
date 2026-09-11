# Testing
## Added flag: `--test_pipeline`

The flag takes patient 10 and 12 (creates the data if not there yet) and runs the pipeline for 1 epoch to check wether the code still runs.

# Preprocessing and augmentation

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
