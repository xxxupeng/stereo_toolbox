from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset
from .utils import pfm_imread

class SceneFlow_Dataset(Stereo_Dataset):
    def load_image_list(self, data_path='/data/xp/Scene_Flow/'):
        if self.data_path is not None:
            data_path = self.data_path

        if self.split == 'train':
            # FlyingThings3D
            self.ref_list = sorted(glob(osp.join(data_path, 'frames_finalpass/TRAIN/*/*/left/*.png')))
            self.tgt_list = [x.replace('left', 'right') for x in self.ref_list]
            self.gt_disp_list = [x.replace('finalpass', 'disparity').replace('.png', '.pfm') for x in self.ref_list]

            # Driving
            self.driving_ref_list = sorted(glob(osp.join(data_path, 'driving_frames_finalpass/*/*/*/left/*.png')))
            self.driving_tgt_list = [x.replace('left', 'right') for x in self.driving_ref_list]
            self.driving_gt_disp_list = [x.replace('frames_finalpass', 'disparity').replace('.png', '.pfm') for x in self.driving_ref_list]

            # Monkaa
            self.monkaa_ref_list = sorted(glob(osp.join(data_path, 'monkaa_frames_finalpass/*/left/*.png')))
            self.monkaa_tgt_list = [x.replace('left', 'right') for x in self.monkaa_ref_list]
            self.monkaa_gt_disp_list = [x.replace('frames_finalpass', 'disparity').replace('.png', '.pfm') for x in self.monkaa_ref_list]

            # Combine
            self.ref_list += self.driving_ref_list + self.monkaa_ref_list
            self.tgt_list += self.driving_tgt_list + self.monkaa_tgt_list
            self.gt_disp_list += self.driving_gt_disp_list + self.monkaa_gt_disp_list
            
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, 'frames_finalpass/TEST/*/*/left/*.png')))
            self.tgt_list = [x.replace('left', 'right') for x in self.ref_list]
            self.gt_disp_list = [x.replace('finalpass', 'disparity').replace('.png', '.pfm') for x in self.ref_list]
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")


    def load_disparity(self, filename):
        disp, _ = pfm_imread(filename)
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        return disp
    

    def load_noc_mask(self, filename):
        return None
    
