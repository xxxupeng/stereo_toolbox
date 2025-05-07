import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2

from .data_augmentation import *
from .utils import *


class Argoverse_Dataset(Dataset):
    """
    Inputs:
    - split: 'train', 'val', 'test'
    - training: True for training, False for testing
    - root_dir: path to the dataset root directory
    
    Outputs: left image, right image, disparity image, non-occulusion mask, raw left image, raw right image
    - disparity and noc mask are filled with nan if not available.
    """
    def __init__(self, split: str, training: bool, root_dir='/data1/xp/Argoverse/'):
        if not os.path.exists(root_dir):
            print(f"Dataset root directory {root_dir} does not exist. Trying to replace '/data1' with '/data'.")
            root_dir = root_dir.replace('/data1', '/data')
            if not os.path.exists(root_dir):
                raise ValueError(f"Dataset root directory {root_dir} does not exist.")
        
        assert split in ['train', 'val', 'test'], "Invalid split name"
        self.split = split
        self.training = training

        dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f'datasets_lists/argoverse/{self.split}.txt')
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
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.
        return disp
    

    def load_noc_mask(self, filename: str):
        if filename is None:
            return None


    def __getitem__(self, index):
        left_image = self.load_image(self.left_images[index])
        right_image = self.load_image(self.right_images[index])
        disp_image = self.load_disp(self.disp_images[index])
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