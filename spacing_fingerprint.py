#!/usr/bin/env python3
"""Compute a target in-plane/z spacing using nnU-Net's own dataset-fingerprinting rule:
median spacing per axis, except an axis is anisotropic (ratio to the finest axis > 3),
in which case that axis uses a low percentile (default 10th) instead of the median.
See: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation/how-nnunet-works.md

No GPU needed -- reads NIfTI headers only, safe to run directly on the login node:
    python spacing_fingerprint.py
"""
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', type=Path, default=Path('data/segthor_part1'),
                        help="Raw SegTHOR root containing train/.")
    parser.add_argument('--out_dir', type=Path, default=Path('results/segthor/spacing_fingerprint'),
                        help="Where to save the evidence (JSON + per-patient CSV).")
    parser.add_argument('--anisotropy_threshold', type=float, default=3.0,
                        help="nnU-Net's default: an axis is 'highly anisotropic' if its median "
                             "spacing is more than this many times the finest axis's median.")
    parser.add_argument('--low_percentile', type=float, default=10.0,
                        help="nnU-Net's default: percentile used for a highly anisotropic axis, "
                             "instead of the median, to avoid upsampling ultra-thick slices.")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    patient_dirs = sorted(p for p in (args.source_dir / 'train').glob('Patient_*') if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(f'No patient folders found under {args.source_dir / "train"}')

    rows = []
    for patient_dir in patient_dirs:
        ct_path = patient_dir / f'{patient_dir.name}.nii.gz'
        dx, dy, dz = nib.load(str(ct_path)).header.get_zooms()[:3]
        rows.append({'patient': patient_dir.name, 'x': float(dx), 'y': float(dy), 'z': float(dz)})

    axes = ['x', 'y', 'z']
    spacings = {axis: np.array([row[axis] for row in rows], dtype=np.float64) for axis in axes}
    medians = {axis: float(np.median(spacings[axis])) for axis in axes}

    finest_axis = min(medians, key=medians.get)
    finest_median = medians[finest_axis]

    target_spacing = {}
    decisions = {}
    for axis in axes:
        ratio = medians[axis] / finest_median
        if ratio > args.anisotropy_threshold:
            value = float(np.percentile(spacings[axis], args.low_percentile))
            decisions[axis] = (f'anisotropic (ratio={ratio:.3f} > {args.anisotropy_threshold}): '
                               f'using {args.low_percentile:g}th percentile instead of median')
        else:
            value = medians[axis]
            decisions[axis] = f'isotropic-enough (ratio={ratio:.3f} <= {args.anisotropy_threshold}): using median'
        target_spacing[axis] = value

    print(f'Fingerprinted {len(rows)} patients from {args.source_dir}\n')
    print('Per-axis median spacing (mm):', {a: round(medians[a], 4) for a in axes})
    print(f'Finest axis: {finest_axis} (median {finest_median:.4f} mm)\n')
    for axis in axes:
        print(f'  {axis}: {decisions[axis]} -> target = {target_spacing[axis]:.4f} mm')
    print(f'\nRecommended target spacing (x, y, z): '
         f'({target_spacing["x"]:.4f}, {target_spacing["y"]:.4f}, {target_spacing["z"]:.4f}) mm')
    print('(Only x/y feeds --target_spacing today -- the pipeline slices per-2D-axial-slice, '
         'z is not resampled.)')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'source_dir': str(args.source_dir),
        'n_patients': len(rows),
        'anisotropy_threshold': args.anisotropy_threshold,
        'low_percentile': args.low_percentile,
        'per_axis_median_mm': medians,
        'finest_axis': finest_axis,
        'decisions': decisions,
        'target_spacing_mm': target_spacing,
        'per_patient_spacing_mm': rows,
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    csv_path = args.out_dir / 'per_patient_spacing.csv'
    with open(csv_path, 'w') as f:
        f.write('patient,x_mm,y_mm,z_mm\n')
        for row in rows:
            f.write(f'{row["patient"]},{row["x"]},{row["y"]},{row["z"]}\n')

    print(f'\nSaved evidence to {args.out_dir}/ (summary.json, per_patient_spacing.csv)')


if __name__ == '__main__':
    main()
