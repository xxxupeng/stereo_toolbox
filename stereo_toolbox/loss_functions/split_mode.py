import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def split_mode(x, maxdisp=192):
    N, D, H, W = x.shape
    assert D == maxdisp

    index = torch.argmax(x,1,keepdim=True)
    mask = torch.arange(D,dtype=x.dtype,device=x.device).reshape([1,D,1,1]).repeat(N,1,H,W)
    mask2 = torch.arange(D+1,dtype=x.dtype,device=x.device).reshape([1,D+1,1,1]).repeat(N,1,H,W)

    x_diff_r = torch.diff(x,dim=1,prepend=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device),\
                        append=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device))
    x_diff_l = torch.diff(x,dim=1,prepend=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device))
    
    index_r = torch.gt(x_diff_r * torch.gt(mask2,index),0).int()
    index_r = torch.argmax(index_r,1,keepdim=True)-1
    
    index_l = torch.lt(x_diff_l * torch.le(mask,index),0).int()
    index_l = (D-1) - torch.argmax(torch.flip(index_l,[1]),1,keepdim=True)
    mask1 = torch.ge(mask,index_l) * torch.le(mask,index_r)

    r = torch.min(index_r-index,index-index_l)
    mask2 = torch.ge(mask,index-r) * torch.le(mask,index+r)

    valid = torch.abs(2*index-index_r-index_l)<3
    mask = valid * mask1 + ~valid * mask2

    mode = x * mask

    return mode, mask