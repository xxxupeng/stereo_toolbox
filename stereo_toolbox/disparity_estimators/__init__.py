import torch

from .unimodal_disparity_estimator import unimodal_disparity_estimator
from .dominant_modal_disparity_estimator import dominant_modal_disparity_estimator


def softargmax_disparity_estimator(x, maxdisp=192):
    disp = torch.arange(maxdisp,dtype=x.dtype,device=x.device).reshape(1,maxdisp,1,1)
    out = torch.sum(x*disp,1, keepdim=True)
    return out


def argmax_disparity_estimator(x, maxdisp=192):
    out = torch.argmax(x,1, keepdim=True)
    return out