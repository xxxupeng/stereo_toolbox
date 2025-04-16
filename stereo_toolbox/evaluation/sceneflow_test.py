import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from stereo_toolbox.datasets import SceneFlow_Dataset


def sceneflow_test(model, split='test_finalpass', device='cuda:0', show_progress=True):
    """
    return epe / px, 1px 2px 3px ourliers / %
    """
    testdataloader = DataLoader(SceneFlow_Dataset(split=split, training=False),
                                batch_size=1, num_workers=16, shuffle=False, drop_last=False)
    if show_progress:
        testdataloader = tqdm(testdataloader, desc='SceneFlow Evaluation', ncols=100)
    
    model = model.to(device).eval()
    
    metrics = np.zeros(4)

    for idx, (left, right, gt_disp, _, _, _) in enumerate(testdataloader):
        left, right, gt_disp = left.to(device), right.to(device), gt_disp.to(device).squeeze()

        mask = (gt_disp > 0) * (gt_disp < 191)
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
            testdataloader.set_description(f"EPE: {metrics[0]/(idx+1):.4f}, 1px: {metrics[1]/(idx+1):.2f}%, 2px: {metrics[2]/(idx+1):.2f}%, 3px: {metrics[3]/(idx+1):.2f}%")

    return metrics / (idx + 1)

        

        



