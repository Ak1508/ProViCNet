import nibabel as nib
import numpy as np

img = nib.load('vcu_TRUS/trus.nii.gz')
mask = nib.load('vcu_TRUS/trus_prostate.nii.gz')

# Rotate mask 90 degrees to match image
new_mask_data = np.rot90(mask.get_fdata(), k=1, axes=(0, 1))

# Save with the EXACT same affine as the image
fixed_mask = nib.Nifti1Image(new_mask_data.astype(np.uint8), img.affine)
nib.save(fixed_mask, 'vcu_TRUS/aligned_prostate.nii.gz')
print("Alignment fixed!")