from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset
from .utils import pfm_imread


class MiddEval3_Dataset(Stereo_Dataset):
    def __init__(self, 
            data_path=None,
            training=True,
            split='train',
            requests=['ref', 'tgt', 'gt_disp'],
            aug_params = {},
            resolution='H'
        ):
        assert resolution in ['Q', 'H', 'F'], "resolution must be one of ['Q', 'H', 'F']"
        self.resolution = resolution

        super().__init__(data_path, training, split, requests, aug_params)


    def load_image_list(self, data_path='/data1/xp/MiddEval3/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/MiddEval3/'
        if not osp.exists(data_path):
            data_path = '/data/xp/MiddEval3/'
            print(f"Warning: {data_path} does not exist. Using '/data/xp/MiddEval3/' instead.")

    
        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, f'training{self.resolution}/*/im0.png')))
            self.tgt_list = [x.replace('im0', 'im1') for x in self.ref_list]
            self.gt_disp_list = [x.replace('im0.png', 'disp0GT.pfm') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, f'test{self.resolution}/*/im0.png')))
            self.tgt_list = [x.replace('im0', 'im1') for x in self.ref_list]
            self.gt_disp_list = [None] * len(self.ref_list)
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        if filename is None:
            return None
        
        disp, _ = pfm_imread(filename)
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        disp[disp == float('inf')] = 0
        return disp
    

    def load_noc_mask(self, filename):
        if filename is None:
            return None
        
        noc_mask = Image.open(filename.replace('disp0GT.pfm', 'mask0nocc.png'))
        noc_mask = np.array(noc_mask) == 255
        return noc_mask.astype(np.uint8)
