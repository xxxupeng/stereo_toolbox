import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import cmapy
import cv2
from matplotlib import pyplot as plt


LC = np.array(
        [  [0,0.0625,49,54,149],
        [0.0625,0.125,69,117,180],
        [0.125,0.25,116,173,209],
        [0.25,0.5,171,217,233],
        [0.5,1,224,243,248],
        [1,2,254,224,144],
        [2,4,253,174,97],
        [4,8,244,109,67],
        [8,16,215,48,39],
        [16,1000000000.0,165,0,38]  ])

def colored_error_map_KITTI(pred, gt, save_file=None, maxdisp=192, threshold=3.):
    if isinstance(pred, torch.Tensor):
        if pred.is_cuda:
            pred = pred.cpu()
        pred = pred.numpy()
    elif not isinstance(pred, np.ndarray):
        raise TypeError("pred must be a torch.Tensor or numpy.ndarray")
    
    if isinstance(gt, torch.Tensor):
        if gt.is_cuda:
            gt = gt.cpu()
        gt = gt.numpy()
    elif not isinstance(gt, np.ndarray):
        raise TypeError("gt must be a torch.Tensor or numpy.ndarray")
    
    pred = pred.squeeze()
    gt = gt.squeeze()

    assert pred.ndim == 2
    assert gt.ndim == 2

    assert pred.shape == gt.shape
    
    d_error = np.abs(gt - pred)
    n_error = d_error/ threshold

    colored_error = np.zeros([pred.shape[0], pred.shape[1], 3])

    for i in range(10):
        mask = (n_error >= LC[i,0] ) * (n_error < LC[i,1])
        colored_error[mask,:] = LC[i,2:]

    valid = (gt > 0) * (gt < maxdisp-1)
    colored_error[~valid,:] = [0,0,0]

    colored_error = colored_error.astype(np.uint8)
    colored_error = cv2.cvtColor(colored_error, cv2.COLOR_RGB2BGR)

    if save_file is not None:
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        cv2.imwrite(save_file, colored_error)

    return colored_error