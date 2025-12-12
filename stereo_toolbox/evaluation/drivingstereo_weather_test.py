import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from stereo_toolbox.datasets_v2 import DrivingStereo_Dataset


def drivingstereo_weather_test(model, device='cuda:0', threshlods = [3, 3, 3, 3], 
                               weathers = ['sunny', 'cloudy', 'rainy', 'foggy'], split='test', resolution='H',
                               maxdisp=192, write_ckpt=None, write_key='ds_weather_test'):
    """
    Generalization evaluation on different weathers of DrivingStereo test sets.
    Outliers threshold: >3px.
    Return [[epe, outliers], % sunny
            [epe, outliers], % cloudy
            [epe, outliers], % rainy
            [epe, outliers]] % foggy       
    """
    model = model.to(device).eval()

    metrics = np.zeros((4, 2))

    for idx, (threshlod, weather) in enumerate(zip(threshlods, weathers)):
        testdataloader = DataLoader(DrivingStereo_Dataset(split=split, training=False,
                                                          requests=['ref','tgt','gt_disp'],
                                                          resolution=resolution,
                                                          weather=weather),
                                    batch_size=1, num_workers=16, shuffle=False, drop_last=False)
        
        image_num = np.zeros(2)
        for data in testdataloader:
            left, right, gt_disp = data['ref'].to(device), data['tgt'].to(device), data['gt_disp'].to(device).squeeze()

            all_mask = (gt_disp > 0) * (gt_disp < maxdisp-1)
            all_num = torch.sum(all_mask).item()

            with torch.no_grad():
                pred_disp = model(left, right).squeeze()
            
            error_map = torch.abs(pred_disp - gt_disp)

            if all_num > 0:
                image_num[0] += 1
                metrics[idx, 0] += error_map[all_mask].mean().item()
            if all_num > 0:
                image_num[1] += 1
                metrics[idx, 1] += (error_map[all_mask] > threshlod).sum().item() / all_num * 100

        metrics[idx] /= image_num

        print(f"DrivingStereo {weather}: "
                f"EPE: {metrics[idx][0]:.4f}px, "
                f"Outliers: {metrics[idx][1]:.4f}%.")

    if write_ckpt:
        checkpoint = torch.load(write_ckpt, map_location='cpu')
        if write_key not in checkpoint:
            checkpoint[write_key] = metrics
            torch.save(checkpoint, write_ckpt)
        else:
            print(f'original generalization metrics:\n{checkpoint[write_key]}')
            print(f'current generalization metrics:\n{metrics}')

    return metrics