from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

from .stereodataset import Stereo_Dataset


class MBS20K_Dataset(Stereo_Dataset):
    def __init__(self, 
            data_path=None,
            training=True,
            split='train',
            requests=['ref', 'tgt', 'gt_disp'],
            aug_params = {},
            baseline_scale=1,
            weather='All',
        ):
        assert baseline_scale in [1,2,3,4], "baseline_scale must be 1, 2, 3, or 4"
        self.baseline_scale = baseline_scale

        assert weather in ['All', 'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNoon', 'MidRainyNoon'], "weather must be one of 'All', 'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNoon', or 'MidRainyNoon'"
        self.weather = weather

        super().__init__(data_path, training, split, requests, aug_params)
        

    def load_image_list(self, data_path='/data1/xp/Carla/data6/'):
        if self.data_path is not None:
            data_path = self.data_path

        # 判断 data_path 是否存在，如果不存在则改为'/data/xp/Carla/data6/'
        if not osp.exists(data_path):
            data_path = '/data/xp/Carla/data6/'
            print(f"Warning: {data_path} does not exist. Using '/data/xp/Carla/data6/' instead.")

        if self.split == 'train':
            self.ref_list = sorted(
                [x for x in glob(osp.join(data_path, f"Town*/{self.weather if self.weather != 'All' else '*'}/*/rgb_0.png")) if 'Town10' not in x]
            )
            self.tgt_list = [x.replace('rgb_0', f'rgb_{self.baseline_scale}') for x in self.ref_list]
            self.gt_disp_list = [x.replace('rgb_0', 'depth_0') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, f"Town10*/{self.weather if self.weather != 'all' else '*'}/*/rgb_0.png")))
            self.tgt_list = [x.replace('rgb_0', f'rgb_{self.baseline_scale}') for x in self.ref_list]
            self.gt_disp_list = [x.replace('rgb_0', 'depth_0') for x in self.ref_list]
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")
        

    def load_disparity(self, filename, focal_length=480, baseline=0.5):
        depth = np.array(Image.open(filename).convert('RGB'))
        depth = (depth @ [1, 256, 256**2]) * 1000 / (256**3 - 1)
        depth[depth > 200] = -1 # ignore depth > 200m

        with np.errstate(divide='ignore', invalid='ignore'):
            disp = (focal_length * baseline * self.baseline_scale) / depth
            disp[~np.isfinite(disp)] = 0  # Replace NaN and inf with 0
        return disp
    
        
    def load_noc_mask(self, filename):
        return None
