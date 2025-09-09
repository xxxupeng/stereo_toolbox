from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset


class KITTI2015_Dataset(Stereo_Dataset):
    def load_image_list(self, data_path='/data/xp/KITTI_2015/'):
        if self.data_path is not None:
            data_path = self.data_path

        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, 'training/image_2/*_10.png')))
            self.tgt_list = [x.replace('image_2', 'image_3') for x in self.ref_list]
            self.gt_disp_list = [x.replace('image_2', 'disp_occ_0') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, 'testing/image_2/*_10.png')))
            self.tgt_list = [x.replace('image_2', 'image_3') for x in self.ref_list]
            self.gt_disp_list = [None] * len(self.ref_list)
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.
        return disp
    

    def load_noc_mask(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename.replace('disp_occ_0', 'disp_noc_0'))
        noc_mask = np.array(disp, dtype=np.float32) > 0
        return noc_mask.astype(np.uint8)
    


class KITTI2012_Dataset(Stereo_Dataset):
    def load_image_list(self, data_path='/data/xp/KITTI_2012/'):
        if self.data_path is not None:
            data_path = self.data_path

        if self.split == 'train':
            self.ref_list = sorted(glob(osp.join(data_path, 'training/colored_0/*_10.png')))
            self.tgt_list = [x.replace('colored_0', 'colored_1') for x in self.ref_list]
            self.gt_disp_list = [x.replace('colored_0', 'disp_occ') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, 'testing/colored_0/*_10.png')))
            self.tgt_list = [x.replace('colored_0', 'colored_1') for x in self.ref_list]
            self.gt_disp_list = [None] * len(self.ref_list)
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename)
        disp = np.array(disp, dtype=np.float32) / 256.
        return disp
    

    def load_noc_mask(self, filename):
        if filename is None:
            return None
        
        disp = Image.open(filename.replace('disp_occ', 'disp_noc'))
        noc_mask = np.array(disp, dtype=np.float32) > 0
        return noc_mask.astype(np.uint8)
