import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from .utils import pfm_imread
from .augmentor import StereoAugmentor


class Stereo_Dataset(Dataset):
    """
    Stereo_Dataset is a PyTorch Dataset class designed for handling stereo image datasets. 
    It provides functionalities for loading stereo image pairs, ground truth disparity maps, 
    and other related data, with optional data augmentation.
    Attributes:
        data_path (str): Path to the dataset.
        training (bool): Indicates whether the dataset is used for training or testing.
        split (str): Dataset split, e.g., 'train', 'val', or 'test'.
        requests (list): List of requested data types to return, such as 'ref', 'tgt', 'gt_disp', ''noc_mask', 'raw_ref', 'raw_tgt', 'ref_filename', 'top_pad', 'right_pad'.
        aug_params (dict): Parameters for data augmentation.
    TBD Methods:
        load_image_list(): Loads the list of image file paths.
        load_dispairty(filename): Loads the ground truth disparity map from a file.
        load_noc_mask(index): Loads the non-occluded mask for the disparity map.
    __getitem__(index):
        Retrieves and processes a data sample at the specified index, applying augmentations if in training mode.
        Returns a dictionary containing the requested data types:
            - ref (torch.Tensor): Reference image in C*H*W format, values in [0, 1].
            - tgt (torch.Tensor): Target image in C*H*W format, values in [0, 1].
            - gt_disp (torch.Tensor): Ground truth disparity map in H*W format, with 0 indicating invalid pixels.
            - noc_mask (torch.Tensor): Non-occluded mask in H*W format, with 0 for occluded and 1 for non-occluded pixels.
            - raw_ref (torch.Tensor): Unaugmented reference image in C*H*W format, values in [0, 1].
            - raw_tgt (torch.Tensor): Unaugmented target image in C*H*W format, values in [0, 1].
            - ref_filename (str): Filename of the reference image.
            - top_pad (int): Number of pixels padded at the top during testing.
            - right_pad (int): Number of pixels padded on the right during testing.
    Notes:
        - The augmentor applies different augmentation strategies during training and testing.
        - During testing, padding is applied to the images, and other augmentations are disabled.
        - The returned data is converted to PyTorch tensors, with images in C*H*W or H*W format.
    """

    def __init__(
            self,
            data_path=None,
            training=True,
            split='train',
            requests=['ref', 'tgt', 'gt_disp'],
            aug_params = {}
        ):
        self.data_path = data_path
        self.split = split
        self.training = training
        self.requests = requests
        
        self.load_image_list()

        if self.training:
            self.augmentor = StereoAugmentor(**aug_params)
        # pad only for test
        else:
            self.augmentor = StereoAugmentor( 
                **(
                    aug_params | {
                        'crop_prob': 0,
                        'color_aug_prob': 0,
                        'color_asym_prob': 0,
                        'spatial_aug_prob': 0,
                        'stretch_prob': 0,
                        'v_flip_prob': 0,
                        'eraser_prob': 0,
                        'pad_prob': 1,
                    }
                )
            )
        

    def load_image_list(self, data_path):
        raise NotImplementedError
    

    def load_dispairty(self, filename):
        raise NotImplementedError
    

    def load_noc_mask(self, index):
        raise NotImplementedError
    

    def load_image(self, filename):
        return np.array(Image.open(filename).convert('RGB')).astype(np.uint8)


    def __len__(self):
        return len(self.ref_list)
    

    def __getitem__(self, index):
        data = {
            'ref': self.load_image(self.ref_list[index]),
            'tgt': self.load_image(self.tgt_list[index])
        }

        if 'gt_disp' in self.requests:
            data['gt_disp'] = self.load_disparity(self.gt_disp_list[index])
        if 'noc_mask' in self.requests:
            data['noc_mask'] = self.load_noc_mask(self.gt_disp_list[index])
        if 'raw_ref' in self.requests:
            data['raw_ref'] = data['ref'].copy()
        if 'raw_tgt' in self.requests:
            data['raw_tgt'] = data['tgt'].copy()

        aug_data = self.augmentor(**data)
        for k in aug_data:
            if aug_data[k] is not None:
                if len(aug_data[k].shape) == 3:
                    aug_data[k] = torch.from_numpy(aug_data[k]).permute(2, 0, 1).float()
                else:
                    aug_data[k] = torch.from_numpy(aug_data[k]).float()
            else:
                del aug_data[k]

        if 'ref_filename' in self.requests:
            aug_data['ref_filename'] = self.ref_list[index]

        return aug_data

            

