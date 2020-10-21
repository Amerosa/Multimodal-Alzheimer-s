import nibabel as nib
import matplotlib.pyplot as plt

from torchvision.transforms import Resize
from PIL import Image

from skimage.transform import resize


MRI_GREY = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\\t1\spm\dartel\group-MultiMode\sub-ADNI005S0221_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-on_probability.nii.gz'
MRI_8 = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\\t1\spm\dartel\group-MultiMode\sub-ADNI005S0221_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-8mm_probability.nii.gz'
#PET_NORM = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\pet\preprocessing\group-MultiMode\sub-ADNI005S0221_ses-M00_task-rest_acq-fdg_pet_space-Ixi549Space_pet.nii.gz'
PET_MASK = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\pet\preprocessing\group-MultiMode\sub-ADNI005S0221_ses-M00_task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_pet.nii.gz'
#PET_PONS = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\pet\preprocessing\group-MultiMode\sub-ADNI005S0221_ses-M00_task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_pet.nii.gz'
PET_8 = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\pet\preprocessing\group-MultiMode\sub-ADNI005S0221_ses-M00_task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_fwhm-8mm_pet.nii.gz'

MRI_CSF = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\\t1\spm\dartel\group-MultiMode\sub-ADNI005S0221_ses-M00_T1w_segm-csf_space-Ixi549Space_modulated-on_probability.nii.gz'
MRI_WHITE = 'F:\caps\subjects\sub-ADNI005S0221\ses-M00\\t1\spm\dartel\group-MultiMode\sub-ADNI005S0221_ses-M00_T1w_segm-whitematter_space-Ixi549Space_modulated-on_probability.nii.gz'

import numpy as np

"""
for path in paths:
    sample = nib.load(path)
    sample = sample.get_fdata()
    #sample = Image.fromarray( np.uint8( (sample[:,sample.shape[1]//2,:])*255 ))
    #sample = sample.resize((228,228))
    sample = sample[:, sample.shape[1]//2,:]
    sample = resize(sample, (228,228))
    plt.imshow(sample, cmap='bone')
    plt.show()
sample = nib.load(PET_8).get_fdata()
sample = np.swapaxes(sample, 0, 1)
from skimage.util import montage

fig, ax1 = plt.subplots(1,1, figsize=(20,20))
ax1.imshow(montage(sample), cmap='bone')
plt.show()

"""
import glob
from shutil import copy



def copy_paths(iterator, dst):
    """
    Takes in a list of paths and copies each obj in that path to destination
    """
    print(f'Copying files to {dst}')
    for path in iterator:
        copy(path, dst)
    print('Finished copying!')
    print('')

mri_greymatter_paths_unfiltered = glob.glob('F:\caps\subjects\**\dartel\**\*T1w_segm-graymatter_space-Ixi549Space_modulated-on_probability.nii.gz', recursive=True)
mri_greymatter_paths_filtered = glob.glob('F:\caps\subjects\**\dartel\**\*T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-8mm_probability.nii.gz', recursive=True)
pet_paths_unfiltered = glob.glob('F:\caps\subjects\**\pet\**\*task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_pet.nii.gz', recursive=True)
pet_paths_filtered = glob.glob('F:\caps\subjects\**\pet\**\*task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_fwhm-8mm_pet.nii.gz', recursive=True)

copy_paths(mri_greymatter_paths_unfiltered, './data/mri/greymatter/unfiltered')
copy_paths(mri_greymatter_paths_filtered,   './data/mri/greymatter/filtered')
copy_paths(pet_paths_unfiltered,            './data/pet/unfiltered')
copy_paths(pet_paths_filtered,              './data/pet/filtered')

