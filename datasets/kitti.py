import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from .data_augmentation import *
from .utils import *


class KITTI_Dataset(Dataset):
    def __init__(self, split: str, training: bool, raw_data:bool=False, root_dir='/data/xp/KITTI_2015/'):
        assert split in ['train', 'train_all', 'val', 'test'], "Invalid split name"
        self.split = split
        self.training = training
        self.raw_data = raw_data

        year = re.findall(r'\d+', root_dir)[0]
        dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f'datasets_lists/kitti{year}/{self.split}.txt')
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
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.
        return disp > 0


    def __getitem__(self, index):
        left_image = self.load_image(self.left_images[index])
        right_image = self.load_image(self.right_images[index])
        disp_image = self.load_disp(self.disp_images[index])
        if disp_image is not None:
            mask_image = self.load_noc_mask(self.disp_images[index].replace('occ', 'noc'))
        else:
            mask_image = self.load_noc_mask(None)

        if self.training:
            left_image, right_image, disp_image, mask_image = random_crop(left_image, right_image, disp_image, mask_image)
            if self.raw_data:
                raw_left_image = transforms.ToTensor()(left_image)
                raw_right_image = transforms.ToTensor()(right_image)
            left_image, right_image = random_jitter(left_image, right_image)
            right_image = random_mask(right_image)
        else:
            left_image, right_image, disp_image, mask_image = pad_to_2x(left_image, right_image, disp_image, mask_image)
            if self.raw_data:
                raw_left_image = transforms.ToTensor()(left_image)
                raw_right_image = transforms.ToTensor()(right_image)
        
        left_image = self.get_transform(left_image)
        right_image = self.get_transform(right_image)

        if disp_image is not None:
            disp_image = torch.from_numpy(np.ascontiguousarray(disp_image)).float()
        else:
            disp_image = torch.zeros(left_image.shape[1:], dtype=left_image.dtype, device=left_image.device)
        
        if mask_image is not None:
            mask_image = torch.from_numpy(np.ascontiguousarray(mask_image)).float()
        else:
            mask_image = torch.ones(left_image.shape[1:], dtype=left_image.dtype, device=left_image.device)

        if self.raw_data:
            return left_image, right_image, disp_image, mask_image, raw_left_image, raw_right_image
        else:
            return left_image, right_image, disp_image, mask_image, torch.zeros_like(left_image), torch.zeros_like(right_image)
        

def KITTI2015_Dataset(split: str, training: bool, raw_data:bool=False, root_dir='/data/xp/KITTI_2015/'):
    return KITTI_Dataset(split, training, raw_data, root_dir='/data/xp/KITTI_2015/')


def KITTI2012_Dataset(split: str, training: bool, raw_data:bool=False, root_dir='/data/xp/KITTI_2012/'):
    return KITTI_Dataset(split, training, raw_data, root_dir='/data/xp/KITTI_2012/')

