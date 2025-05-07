from torchvision.transforms import ColorJitter, functional, Compose
import numpy as np
import random
from PIL import Image

color_aug = Compose([
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14),
    lambda img: functional.adjust_gamma(img, random.uniform(0.8, 1.2))
])


def random_jitter(left, right):
    """
    Input format: Image
    Output format: Array
    """
    left = np.array(color_aug(Image.fromarray(left)))
    right = np.array(color_aug(Image.fromarray(right)))
    return left, right


def random_crop(left, right, disp=None, mask=None, crop_size=[384, 512]):
    left = np.array(left)
    right = np.array(right)

    H, W, C = left.shape
    crop_H, crop_W = crop_size
    if crop_H > H: crop_H = H
    if crop_W > W: crop_W = W

    h = random.randint(0, H - crop_H)
    w = random.randint(0, W - crop_W)

    left = left[h : h + crop_H, w : w + crop_W]
    right = right[h : h + crop_H, w : w + crop_W]
    if disp is not None:
        disp = disp[..., h : h + crop_H, w : w + crop_W]
    if mask is not None:
        mask = mask[h : h + crop_H, w : w + crop_W]

    return left, right, disp, mask


def random_mask(right):
    right.flags.writeable = True

    if np.random.binomial(1, 0.5):
        sx = int(np.random.uniform(35,100))
        sy = int(np.random.uniform(25,75))
        cx = int(np.random.uniform(sx, right.shape[0]-sx))
        cy = int(np.random.uniform(sy, right.shape[1]-sy))
        right[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right,0),0)[np.newaxis,np.newaxis]
    
    return right


def pad_to_2x(left, right, disp=None, mask=None):
    left = np.array(left)
    right = np.array(right)

    H, W, C = left.shape

    scale = 96
    top_pad = int(np.ceil(H / scale) * scale - H)
    right_pad = int(np.ceil(W / scale) * scale - W)

    left = np.lib.pad(left, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
    right = np.lib.pad(right, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
    
    if disp is not None:
        if disp.ndim == 2:    # disparity
            disp = np.lib.pad(disp, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        elif disp.ndim == 3:  # distribution
            disp = np.lib.pad(disp, ((0,0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        
    if mask is not None:
        assert disp.ndim == 2
        mask = np.lib.pad(mask, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

    return left, right, disp, mask

