import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from stereo_toolbox.datasets import KITTI2015_Dataset, KITTI2012_Dataset, MiddleburyEval3_Dataset, ETH3D_Dataset


def generalization_eval(model, device='cuda:0', threshlods = [3, 3, 2, 1], splits = ['train_all', 'train_all', 'trainH_all', 'train_all'], maxdisp=192, write_ckpt=None):
    """
    Generalization evaluation on training sets of public datasets.
    Outliers threshold: kitti 2015 >3px; kitti 2012 >3px; middlebury eval3 >2px; eth3d >1px.
    Return [[epe, occ, noc, all], % kitti 2015 
            [epe, occ, noc, all], % kitti 2012
            [epe, occ, noc, all], % middlebury eval3
            [epe, occ, noc, all]] % eth3d       
    """
    model = model.to(device).eval()

    metrics = np.zeros((4, 4))

    for idx, (dataset, threshlod, split) in enumerate(zip([KITTI2015_Dataset, KITTI2012_Dataset, MiddleburyEval3_Dataset, ETH3D_Dataset], threshlods, splits)):
        testdataloader = DataLoader(dataset(split=split, training=False),
                                    batch_size=1, num_workers=16, shuffle=False, drop_last=False)
        
        image_num = np.zeros(4)
        for data in testdataloader:
            left, right, gt_disp, noc_mask = data['left'].to(device), data['right'].to(device), data['gt_disp'].to(device).squeeze(), data['noc_mask'].to(device).squeeze()

            all_mask = (gt_disp > 0) * (gt_disp < maxdisp-1)
            noc_mask = noc_mask.bool() * all_mask
            occ_mask = ~noc_mask * all_mask

            all_num = torch.sum(all_mask).item()
            noc_num = torch.sum(noc_mask).item()
            occ_num = torch.sum(occ_mask).item()

            with torch.no_grad():
                pred_disp = model(left, right).squeeze()
            
            error_map = torch.abs(pred_disp - gt_disp)

            if all_num > 0:
                image_num[0] += 1
                metrics[idx, 0] += error_map[all_mask].mean().item()
            if occ_num > 0:
                image_num[1] += 1
                metrics[idx, 1] += (error_map[occ_mask] > threshlod).sum().item() / occ_num * 100
            if noc_num > 0:
                image_num[2] += 1
                metrics[idx, 2] += (error_map[noc_mask] > threshlod).sum().item() / noc_num * 100
            if all_num > 0:
                image_num[3] += 1
                metrics[idx, 3] += (error_map[all_mask] > threshlod).sum().item() / all_num * 100

        metrics[idx] /= image_num

        print(f"{dataset.__name__} "
                f"EPE: {metrics[idx][0]:.4f}px, "
                f"OCC: {metrics[idx][1]:.4f}%, "
                f"NOC: {metrics[idx][2]:.4f}%, "
                f"ALL: {metrics[idx][3]:.4f}%.")

    if write_ckpt:
        checkpoint = torch.load(write_ckpt, map_location='cpu')
        if 'generalization' not in checkpoint:
            checkpoint['generalization'] = metrics
            torch.save(checkpoint, write_ckpt)
        else:
            print(f'original generalization metrics:\n{checkpoint["generalization"]}')
            print(f'current generalization metrics:\n{metrics}')

    return metrics