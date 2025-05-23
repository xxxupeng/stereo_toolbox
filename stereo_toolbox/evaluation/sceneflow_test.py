import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from stereo_toolbox.datasets import SceneFlow_Dataset


def sceneflow_test(model, split='test_finalpass', device='cuda:0', show_progress=True, maxdisp=192, write_ckpt=None, write_key='sceneflow'):
    """
    return epe / px, 1px 2px 3px ourliers / %
    """
    testdataloader = DataLoader(SceneFlow_Dataset(split=split, training=False),
                                batch_size=1, num_workers=16, shuffle=False, drop_last=False)
    if show_progress:
        testdataloader = tqdm(testdataloader, desc='SceneFlow Evaluation', ncols=100)
    
    model = model.to(device).eval()
    
    metrics = np.zeros(4)

    for idx, data in enumerate(testdataloader):
        left, right, gt_disp = data['left'].to(device), data['right'].to(device), data['gt_disp'].to(device).squeeze()

        mask = (gt_disp > 0) * (gt_disp < maxdisp-1)
        valid_num = mask.sum().item()
        if valid_num == 0:
            continue

        with torch.no_grad():
            pred_disp = model(left, right).squeeze()

        error_map = torch.abs(pred_disp - gt_disp)[mask]

        metrics[0] += error_map.mean().item()
        for outlier in [1, 2, 3]:
            metrics[outlier] += (error_map > outlier).sum().item() / valid_num * 100

        # update tqdm desc
        if show_progress:
            testdataloader.set_description(f"EPE: {metrics[0]/(idx+1):.4f}px, 1px: {metrics[1]/(idx+1):.4f}%, 2px: {metrics[2]/(idx+1):.4f}%, 3px: {metrics[3]/(idx+1):.4f}%")

    metrics = metrics / (idx + 1)

    if write_ckpt:
        checkpoint = torch.load(write_ckpt, map_location='cpu')
        if write_key not in checkpoint:
            checkpoint[write_key] = metrics
            torch.save(checkpoint, write_ckpt)
        else:
            print(f'original sceneflow metrics:\n{checkpoint[write_key]}')
            print(f'current sceneflow metrics:\n{metrics}')

    return metrics

        

        



