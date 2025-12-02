from PIL import Image
import numpy as np
from glob import glob
import os.path as osp
import tifffile as tiff
import numpy as np
import re

from .stereodataset import Stereo_Dataset


class SCARED_Dataset(Stereo_Dataset):
    def load_image_list(self, data_path='/data1/xp/SCARED/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/SCARED/'
        if not osp.exists(data_path):
            print(f"Warning: {data_path} does not exist. Using '/data/xp/SCARED/' instead.")
            data_path = '/data/xp/SCARED/'

        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, 'dataset_*/keyframe_*/Left_Image.png')))
            self.tgt_list = [x.replace('Left_Image', 'Right_Image') for x in self.ref_list]
            self.gt_disp_list = [x.replace('Left_Image.png', 'left_depth_map.tiff') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, 'test_dataset_*/keyframe_*/Left_Image.png')))
            self.tgt_list = [x.replace('Left_Image', 'Right_Image') for x in self.ref_list]
            self.gt_disp_list = [None] * len(self.ref_list)
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def _extract_matrix_data(self, content, key):
        pattern = rf"{key}:[^\[]*data:\s*\[([^\]]+)\]"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if not match:
            raise ValueError(f"Expected matrix data for {key}")
        tokens = [token for token in re.split(r"[,\s]+", match.group(1)) if token]
        return [float(token) for token in tokens]
    

    def load_disparity(self, filename):
        if filename is None:
            return None
        
        depth = tiff.imread(filename)[..., 2]

        calibration_path = filename.replace('left_depth_map.tiff', 'endoscope_calibration.yaml')
        with open(calibration_path, "r") as f:
            calibration_text = f.read()
        t_values = self._extract_matrix_data(calibration_text, "T")
        baseline = abs(t_values[0])
        m1_values = self._extract_matrix_data(calibration_text, "M1")
        focal_length = m1_values[0]

        epsilon = 1e-8
        disp = (baseline * focal_length) / (depth + epsilon)
        disp[~np.isfinite(disp)] = 0

        return disp


    def load_noc_mask(self, filename):
        return None
    
