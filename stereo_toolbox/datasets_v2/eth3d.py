from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset
from .utils import pfm_imread


class ETH3D_Dataset(Stereo_Dataset):
    def __init__(self, 
            data_path=None,
            training=True,
            split='train',
            requests=['ref', 'tgt', 'gt_disp'],
            aug_params = {},
        ):

        super().__init__(data_path, training, split, requests, aug_params)


    def load_image_list(self, data_path='/data1/xp/ETH3D/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/ETH3D/'
        if not osp.exists(data_path):
            print(f"Warning: {data_path} does not exist. Using '/data/xp/ETH3D/' instead.")
            data_path = '/data/xp/ETH3D/'
    
        self.ref_list = sorted(glob(osp.join(data_path, f'*/im0.png')))
        self.tgt_list = [x.replace('im0', 'im1') for x in self.ref_list]
            

        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, f'*/im0.png')))
            self.ref_list = [x for x in self.ref_list if osp.exists(x.replace('im0.png', 'disp0GT.pfm'))]
            self.tgt_list = [x.replace('im0', 'im1') for x in self.ref_list]
            self.gt_disp_list = [x.replace('im0.png', 'disp0GT.pfm') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, f'*/im0.png')))
            self.ref_list = [x for x in self.ref_list if not osp.exists(x.replace('im0.png', 'disp0GT.pfm'))]
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
