#!/usr/bin/env python3
"""One-off: slice Patient_07's CT, GT, and GT2 into PNGs for viewer.py comparison.
Run on Snellius (has nibabel/skimage already):
    python slice_patient07_gt2.py
Then scp the whole --dest folder down and run viewer.py locally (see readme.md's
2D viewer section) on data/patient07_compare/{img,gt,gt2}.
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.io import imsave

from slice_segthor import norm_arr, resize_


def main(source_dir: Path, dest_dir: Path, shape: tuple[int, int]) -> None:
    patient_dir = source_dir / 'train' / 'Patient_07'
    ct = np.asarray(nib.load(str(patient_dir / 'Patient_07.nii.gz')).dataobj)
    gt = np.asarray(nib.load(str(patient_dir / 'GT.nii.gz')).dataobj)
    gt2 = np.asarray(nib.load(str(patient_dir / 'GT2.nii.gz')).dataobj)

    norm_ct = norm_arr(ct)
    _, _, z = ct.shape

    for subfolder, volume, is_mask in [('img', norm_ct, False), ('gt', gt, True), ('gt2', gt2, True)]:
        out_dir = dest_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        for idz in range(z):
            order = 0 if is_mask else 1
            sl = resize_(volume[:, :, idz], shape, order=order).astype(np.uint8)
            if is_mask:
                sl = sl * 63
            imsave(str(out_dir / f'Patient_07_{idz:04d}.png'), sl)

    print(f'Wrote {z} slices each to {dest_dir}/img, /gt, /gt2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=Path, default=Path('data/segthor_part1'))
    parser.add_argument('--dest_dir', type=Path, default=Path('data/patient07_compare'))
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256])
    args = parser.parse_args()
    main(args.source_dir, args.dest_dir, tuple(args.shape))
