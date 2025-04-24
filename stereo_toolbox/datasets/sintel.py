import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from .data_augmentation import *
from .utils import *


class Sintel_Dataset(Dataset):
    """
    Inputs:
    - split: 'train_clean', 'train_final'
    - training: True for training, False for testing
    - root_dir: path to the dataset root directory
    
    Outputs: left image, right image, disparity image, non-occulusion mask, raw left image, raw right image
    - disparity and noc mask are filled with nan if not available.
    """
    def __init__(self, split: str, training: bool, root_dir='/data/xp/Sintel/'):
        assert split in ['train_clean', 'train_final'], "Invalid split name"
        self.split = split
        self.training = training

        dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f'datasets_lists/sintel/{self.split}.txt')
        self.left_images, self.right_images, self.disp_images = read_lines(dataset_file)

        self.left_images = [os.path.join(root_dir, line) for line in self.left_images]
        self.right_images = [os.path.join(root_dir, line) for line in self.right_images]
        if self.disp_images[0] is not None:
            self.disp_images = [os.path.join(root_dir, line) for line in self.disp_images]

        self.get_transform = get_transform()


    def __len__(self):
        return len(self.left_images)
    

    def load_image(self, filename: str):
        return Image.open(filename).convert('RGB')
    

    def load_disp(self, filename: str):
        if filename is None:
            return None
        
        disp = np.array(Image.open(filename), dtype=np.float32)
        disp = disp[...,0] * 4 + disp[...,1] / (2**6) + disp[...,2] / (2**14)
        return disp
    

    def load_noc_mask(self, filename: str):
        if filename is None:
            return None
        
        noc_mask = Image.open(filename).convert('L')
        noc_mask = np.array(noc_mask, dtype=np.uint8)
        
        return noc_mask == 0


    def __getitem__(self, index):
        left_image = self.load_image(self.left_images[index])
        right_image = self.load_image(self.right_images[index])
        disp_image = self.load_disp(self.disp_images[index])
        if disp_image is not None:
            mask_image = self.load_noc_mask(self.disp_images[index].replace('disparities', 'occlusions'))
        else:
            mask_image = self.load_noc_mask(None)

        if self.training:
            left_image, right_image, disp_image, mask_image = random_crop(left_image, right_image, disp_image, mask_image)
            raw_left_image = transforms.ToTensor()(left_image)
            raw_right_image = transforms.ToTensor()(right_image)
            left_image, right_image = random_jitter(left_image, right_image)
            right_image = random_mask(right_image)
        else:
            left_image, right_image, disp_image, mask_image = pad_to_2x(left_image, right_image, disp_image, mask_image)
            raw_left_image = transforms.ToTensor()(left_image)
            raw_right_image = transforms.ToTensor()(right_image)
        
        left_image = self.get_transform(left_image)
        right_image = self.get_transform(right_image)

        if disp_image is not None:
            disp_image = torch.from_numpy(np.ascontiguousarray(disp_image)).float()
        else:
            disp_image = torch.full(left_image.shape[1:], float('nan'), dtype=left_image.dtype, device=left_image.device)
        
        if mask_image is not None:
            mask_image = torch.from_numpy(np.ascontiguousarray(mask_image)).float()
        else:
            mask_image = torch.full(left_image.shape[1:], float('nan'), dtype=left_image.dtype, device=left_image.device)

        return {
            'left': left_image,
            'right': right_image,
            'gt_disp': disp_image,
            'noc_mask': mask_image,
            'raw_left': raw_left_image,
            'raw_right': raw_right_image
        }        