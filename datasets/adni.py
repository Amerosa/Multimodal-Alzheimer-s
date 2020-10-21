import os 
import csv
import torch

import torchvision.utils as v_utils
import torchvision.transforms as standard_transforms

from torch.utils.data import DataLoader, Dataset, random_split

import utils.adni_utils as a_utils 

from PIL import Image
from skimage.transform import resize as sk_resize

import matplotlib.pyplot as plt

import numpy as np
import glob
import nibabel as nib 

class Greymatter(Dataset):
    def __init__(self, seg, filtered_flag=False, input_transform=None, target_transform=None):

        assert seg in ['greymatter', 'whitematter', 'csf'], 'Please selected a correct option for the type of MRI segmentaiton'
        
        filt = 'filtered' if filtered_flag else 'unfiltered'
        mri_path = os.path.join('../data', 'mri', seg, filt, '*.nii.gz') 
        pet_path = os.path.join('../data', 'pet', filt, '*.nii.gz') 
        
        self.mri_paths = sorted(glob.glob(mri_path))
        self.pet_paths = sorted(glob.glob(pet_path))
        self.num_slices = 145
        self.mappings = generate_mappings()
        self.input_transform = input_transform
    
    def __len__(self):
        return len(self.mri_paths) * self.num_slices


    def __getitem__(self, index):
            
        slice_index  =  index % self.num_slices
        volume_index =  index // self.num_slices
         
        subject_id, session_id = parse_adni_subject(self.mri_paths[volume_index])
        key = '_'.join([subject_id, session_id])


        print(subject_id, session_id, slice_index)
        print(f'Volume: {volume_index} Slice: {slice_index}')
        mri_image = np.array(nib.load(self.mri_paths[volume_index]).get_fdata())
        print(f'Type: {type(mri_image)} Shape: {mri_image.shape}')
        mri_image = Image.fromarray(mri_image[:, slice_index, :])
        
        
        print(f'Volume: {volume_index} Slice: {slice_index}')
        pet_image = np.array(nib.load(self.pet_paths[volume_index]).get_fdata())
        print(f'Type: {type(pet_image)} Shape: {pet_image.shape}')
        pet_image = Image.fromarray(pet_image[:, slice_index, :])
        

        if self.input_transform is not None:
            mri_image = self.input_transform(mri_image)
            pet_image = self.input_transform(pet_image)


        return (mri_image, pet_image, key, self.mappings[key], slice_index)


def parse_adni_subject(pth):
    x = pth.split('\\')[-1]
    return x.split('_')[:2]

class GreymatterLoader:
    def __init__(self):
        
        input_transform = standard_transforms.Compose([
            standard_transforms.Resize((244,244)),
            standard_transforms.ToTensor(),
            #standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Lambda(lambda x: x.expand(3,-1,-1))
            ])

        #TODO NEED TO ADD TARGET TRANSFORM AND THE REVERSERS OF THESE

        adni_dataset = Greymatter(seg='greymatter', filtered_flag=False, input_transform=input_transform)
        print(len(adni_dataset))
        train_size = int(len(adni_dataset)*0.6)
        val_size = int(len(adni_dataset)*0.2)
        test_size = int(len(adni_dataset) - (train_size+val_size))
        assert len(adni_dataset) == train_size + val_size + test_size, 'Whoops dataset split wrong'
        train_set, val_set, test_set = random_split(adni_dataset, [train_size, val_size, test_size])
        
        #TODO need to update all config properties once logger and config files set 
        self.train_loader = DataLoader(train_set, batch_size=b, pin_memory=True)
        self.val_loader = DataLoader(val_set, batch_size=b, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=b, pin_memory=True)
        
        print(f'Dataset is {len(adni_dataset)} long')

from torchvision.utils import make_grid

b=4
adni = GreymatterLoader()
#sample = next(iter(adni.train_loader))
#mri_batch, pet_batch, name, label, slc = sample 
#plt.plot(make_grid(mri_batch))
"""
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))        
ax1.imshow(mri_image[0][0], cmap='bone')
ax2.imshow(pet_image[0][0], cmap='bone')
plt.show()
   
"""



