from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset


class DrivingStereo_Dataset(Stereo_Dataset):
    """
    split weather resolution options:
    - train None H
    - test  None H
    - test  None F
    - test  sunny H
    - test  sunny F
    - test  rainy H
    - test  rainy F
    - test  foggy H
    - test  foggy F
    - test  cloudy H
    - test  cloudy F
    """
    def __init__(self, 
            data_path=None,
            training=True,
            split='test',
            requests=['ref', 'tgt', 'gt_disp'],
            aug_params = {},
            resolution='H',
            weather = 'sunny',
        ):
        assert resolution in ['H', 'F'], "resolution must be one of ['H', 'F']"
        assert weather in ['', 'foggy', 'rainy', 'sunny', 'cloudy'], "weather must be one of ['', 'foggy', 'rainy', 'sunny', 'cloudy']"

        if weather == '' and split == 'train':
            assert resolution == 'H', "For training split without weather condition, resolution must be 'H'"
        if weather != '':
            assert split == 'test', "When weather condition is specified, split must be 'test'"

        self.image_format = 'jpg' if resolution == 'H' else 'png'
        self.resolution = 'half' if resolution == 'H' else 'full'
        self.weather = weather

        super().__init__(data_path, training, split, requests, aug_params)


    def load_image_list(self, data_path='/data1/xp/Driving_Stereo/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/Driving_Stereo/'
        if not osp.exists(data_path):
            print(f"Warning: {data_path} does not exist. Using '/data/xp/Driving_Stereo/' instead.")
            data_path = '/data/xp/Driving_Stereo/'

        if self.weather != '':
            if self.split in ['train', 'test']:
                self.ref_list = sorted(glob(osp.join(data_path, f'{self.weather}/left-image-{self.resolution}-size/*.{self.image_format}')))
                self.tgt_list = [x.replace('left-image', 'right-image') for x in self.ref_list]
                self.gt_disp_list = [x.replace('left-image', 'disparity-map').replace('.jpg', '.png') for x in self.ref_list]
            else:
                raise ValueError(f"split must be 'train', not {self.split}")
        elif self.weather == '':
            if self.split == 'train':
                self.ref_list = sorted(glob(osp.join(data_path, f'{self.split}-left-image/*/*.{self.image_format}')))
                self.tgt_list = [x.replace('left-image', 'right-image') for x in self.ref_list]
                self.gt_disp_list = [x.replace('left-image', 'disparity-map').replace('.jpg', '.png') for x in self.ref_list]
            elif self.split == 'test':
                self.ref_list = sorted(glob(osp.join(data_path, f'{self.split}-left-image/left-image-{self.resolution}-size/*/*.{self.image_format}')))
                self.tgt_list = [x.replace('left-image', 'right-image').replace('left-image', 'right-image') for x in self.ref_list]
                self.gt_disp_list = [x.replace('left-image', 'disparity-map').replace('left-image', 'disparity-map').replace('.jpg', '.png') for x in self.ref_list]
            else:
                raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.
        return disp
    

    def load_noc_mask(self, filename):
        return None

    