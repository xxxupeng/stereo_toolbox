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
                [x for x in glob(osp.join(data_path, f"Town*/{self.weather if self.weather.lower() != 'all' else '*'}/*/rgb_0.png")) if 'Town10' not in x]
            )
            self.tgt_list = [x.replace('rgb_0', f'rgb_{self.baseline_scale}') for x in self.ref_list]
            self.gt_disp_list = [x.replace('rgb_0', 'depth_0') for x in self.ref_list]
        elif self.split == 'test':
            self.ref_list = sorted(glob(osp.join(data_path, f"Town10*/{self.weather if self.weather.lower() != 'all' else '*'}/*/rgb_0.png")))
            self.tgt_list = [x.replace('rgb_0', f'rgb_{self.baseline_scale}') for x in self.ref_list]
            self.gt_disp_list = [x.replace('rgb_0', 'depth_0') for x in self.ref_list]
        else:
            raise ValueError(f"split must be 'train' or 'test', not {self.split}")
        

    def load_disparity(self, filename, focal_length=480, baseline=0.5):
        depth_rgb = np.asarray(Image.open(filename).convert('RGB'), dtype=np.float32)
        depth = (depth_rgb @ np.array([1.0, 256.0, 256.0**2], dtype=np.float32)) * (1000.0 / (256.0**3 - 1.0))

        # Depth encoding is 0~1000 where 1000 is infinity.
        # Only keep valid metric depth in (0, 200]; set others to zero disparity.
        valid = (depth > 0.0) & (depth <= 200.0)
        disp = np.zeros_like(depth, dtype=np.float32)
        scale = float(focal_length * baseline * self.baseline_scale)
        disp[valid] = scale / depth[valid]
        return disp
    
        
    def load_noc_mask(self, filename):
        return None
