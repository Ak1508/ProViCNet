import nibabel as nib
mask = nib.load('vcu_TRUS/trus_prostate.nii.gz')
img = nib.load('vcu_TRUS/trus.nii.gz')
print(img.shape, mask.shape)