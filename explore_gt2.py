"""Interactive exploration of Patient_07's GT vs GT2 mislabeling.
Run in VS Code as a Jupyter Interactive Window: open this file, use the
"Run Cell" links that appear above each '# %%' marker (or Shift+Enter).
"""
# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

patient_dir = 'data/segthor_part1/train/Patient_07'
ct = np.asarray(nib.load(f'{patient_dir}/Patient_07.nii.gz').dataobj)
gt = np.asarray(nib.load(f'{patient_dir}/GT.nii.gz').dataobj)
gt2 = np.asarray(nib.load(f'{patient_dir}/GT2.nii.gz').dataobj)

diff_mask = (gt == 1) & (gt2 == 4)  # voxels relabeled esophagus -> aorta
z_with_diff = np.where(diff_mask.any(axis=(0, 1)))[0]
print(f'{diff_mask.sum()} differing voxels, spanning z-slices {z_with_diff.min()}-{z_with_diff.max()}')

# %% 2D: scroll through slices, CT with GT (left) vs GT2 (right) overlaid
def show_slice(z=int(z_with_diff.mean())):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mask, title in zip(axes, [gt[:, :, z], gt2[:, :, z], diff_mask[:, :, z]],
                               ['GT (esophagus=1 swallows aorta)', 'GT2 (corrected)', 'Differing voxels']):
        ax.imshow(ct[:, :, z], cmap='gray')
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='autumn', alpha=0.6)
        ax.set_title(f'{title}, z={z}')
        ax.axis('off')
    plt.show()

interact(show_slice, z=IntSlider(min=int(z_with_diff.min()), max=int(z_with_diff.max()),
                                 value=int(z_with_diff.mean())))

# %% 3D: shape of the differing region only (subsampled for speed)
xs, ys, zs = np.where(diff_mask)
step = max(1, len(xs) // 20000)  # cap points plotted for responsiveness
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(xs[::step], ys[::step], zs[::step], s=1, alpha=0.3)
ax.set_title('3D shape of relabeled (esophagus->aorta) voxels')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z (slice)')
plt.show()
