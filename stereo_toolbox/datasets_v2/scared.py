from PIL import Image
import numpy as np
from glob import glob
import os.path as osp
import tifffile as tiff
import numpy as np
import re

from .stereodataset import Stereo_Dataset

"""
https://github.com/dimitrisPs/scared_toolkit
please refer to this repo for more details about SCARED dataset.

python -m scripts.generate_keyframe_dataset root_dir [--out_dir] --recursive --depth --undistort --disparity --alpha 0

disparity is saved in 16-bit png format, need to divide by 256 to get the actual disparity values.

Disparity has a scale issue, multiplication by 2 is needed for alignment (not analyzed thoroughly yet).
The largest 1% disparity values are removed to eliminate noise.

"""
class SCARED_Dataset(Stereo_Dataset):
    def load_image_list(self, data_path='/data1/xp/SCARED/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/SCARED/'
        if not osp.exists(data_path):
            print(f"Warning: {data_path} does not exist. Using '/data/xp/SCARED/' instead.")
            data_path = '/data/xp/SCARED/'

        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, 'dataset_*/keyframe_*/left_rectified.png')))
            self.tgt_list = [x.replace('left_rectified', 'right_rectified') for x in self.ref_list]
            self.gt_disp_list = [x.replace('left_rectified.png', 'disparity.png') for x in self.ref_list]
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.

        disp = disp * 2.0  # scale issue in SCARED dataset
        disp[disp > np.percentile(disp, 99)] = 0  # remove the largest 1% disparity values to eliminate noise

        return disp


    def load_noc_mask(self, filename):
        return None
    
