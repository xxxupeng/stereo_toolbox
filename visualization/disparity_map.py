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


def colored_disparity_map_Spectral_r(disp, maxval=0, save_file=None):
    """
    BGR format
    """
    if isinstance(disp, torch.Tensor):
        if disp.is_cuda:
            disp = disp.cpu()
        disp = disp.numpy()
    elif not isinstance(disp, np.ndarray):
        raise TypeError("disp must be a torch.Tensor or numpy.ndarray")
    
    if maxval == 0:
        maxval = np.max(np.where(np.isinf(disp), -np.inf, disp))

    disp = (np.clip(disp.squeeze(), 0, maxval) / maxval * 255.0).astype(np.uint8)
    assert disp.ndim == 2, "disp must be a 2D array"

    colored_disp = cv2.applyColorMap(disp, cmapy.cmap('Spectral_r'))

    if save_file is not None:
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        cv2.imwrite(save_file, colored_disp)

    return colored_disp


def colored_dispairty_map_KITTI(disp, maxval=0, save_file=None):
    """
    BGR format
    """
    if isinstance(disp, torch.Tensor):
        if disp.is_cuda:
            disp = disp.cpu()
        disp = disp.numpy()
    elif not isinstance(disp, np.ndarray):
        raise TypeError("disp must be a torch.Tensor or numpy.ndarray")
    
    if maxval == 0:
        maxval = np.max(np.where(np.isinf(disp), -np.inf, disp))
    minval = 0

    colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
    weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,
                          8.771929824561404,5.405405405405405,8.771929824561404,0])
    cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

    colored_disp = np.zeros([disp.shape[0], disp.shape[1], 3])
    values = np.expand_dims(np.minimum(np.maximum((disp-minval)/(maxval-minval), 0.), 1.), -1)
    bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disp.shape[1], axis=1), disp.shape[0], axis=0)
    diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
    index = np.argmax(diffs, axis=-1)-1

    w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]

    colored_disp[:,:,0] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
    colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
    colored_disp[:,:,2] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

    colored_disp = (colored_disp*np.expand_dims((disp>0),-1)*255).astype(np.uint8)

    # Convert to BGR format
    colored_disp = cv2.cvtColor(colored_disp, cv2.COLOR_RGB2BGR)

    if save_file is not None:
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        cv2.imwrite(save_file, colored_disp)

    return colored_disp
