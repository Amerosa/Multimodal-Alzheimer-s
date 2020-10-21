from pathlib import Path

import utils.adni_utils as a_utils

from pprint import pprint

import torch

import torchvision.utils as v_utils
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image

class Entropy(Dataset):
    def __init__(self, data_root, mode, input_transform=None):
        
        assert mode in ['mri', 'pet', 'both']

        self.data_root = data_root
        self.mode = mode
        self.input_transform = input_transform
        self.mri_paths = list(Path(data_root).rglob('mri/*.png'))
        self.pet_paths = list(Path(data_root).rglob('pet/*.png'))
        self.mappings = a_utils.generate_mappings()

        #pprint(self.mri_paths)
        assert len(self.mri_paths) == len(self.pet_paths)

    def __len__(self):
        return len(self.mri_paths)

    def __getitem__(self, index):

        subj = self.mri_paths[index].parts[-4]
        sess = self.mri_paths[index].parts[-3]

        mri_image = Image.open(self.mri_paths[index])
        pet_image = Image.open(self.pet_paths[index])
        
        if self.input_transform:
            mri_image = self.input_transform(mri_image)
            pet_image = self.input_transform(pet_image)

        label = self.mappings[ '_'.join([subj, sess]) ]

        if self.mode == 'mri':
            return (mri_image, label)

        if self.mode == 'pet':
            return (pet_image, label)

        if self.mode == 'both':
            return (mri_image, pet_image, label)

class EntropyLoader:
    def __init__(self, config, split=True):

        self.config = config

        #Values for normalization were taken form the torchvision documents. Using pre trained resnet.
        input_transform = standard_transforms.Compose([
            standard_transforms.Resize((224,224), interpolation=Image.BILINEAR),
            standard_transforms.ToTensor(),
            #standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                              std=[0.229, 0.224, 0.225]),
            standard_transforms.Lambda(lambda x: x.expand(3,-1,-1))
            ])
        
        entropy_dataset = Entropy(config.data_folder, config.modality, input_transform=input_transform)

        if split:
            train_size = int(len(entropy_dataset)*0.7)
            test_size = int(len(entropy_dataset) - train_size)
            #val_size = int(len(entropy_dataset)*0.2)
            #test_size = int(len(entropy_dataset) - (train_size+val_size))
            #assert len(entropy_dataset) == train_size + val_size + test_size, 'Whoops dataset split wrong'

            assert len(entropy_dataset) == train_size + test_size, 'Whoops dataset split wrong'

            torch.manual_seed(config.seed)
            #train_set, val_set, test_set = random_split(entropy_dataset, [train_size, val_size, test_size])
            train_set, test_set = random_split(entropy_dataset, [train_size, test_size])
            self.train_loader = DataLoader(train_set, batch_size=config.batch_size, pin_memory=True)
            #self.val_loader   = DataLoader(val_set,   batch_size=config.batch_size, pin_memory=True)
            self.test_loader  = DataLoader(test_set,  batch_size=config.batch_size, pin_memory=True)
        else:
            self.full_loader = DataLoader(entropy_dataset, batch_size=self.config.batch_size, pin_memory=True)