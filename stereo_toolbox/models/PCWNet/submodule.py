from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x)))


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def disp_regression_nearby(similarity, disp_step, half_support_window=2):
    """Returns predicted disparity with subpixel_map(disp_similarity).

    Predicted disparity is computed as:

    d_predicted = sum_d( d * P_predicted(d)),
    where | d - d_similarity_maximum | < half_size

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        half_support_window: defines size of disparity window in pixels
                             around disparity with maximum similarity,
                             which is used to convert similarities
                             to probabilities and compute mean.
    """

    assert 4 == similarity.dim(), \
        'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())

    # In every location (x, y) find disparity with maximum similarity score.
    similar_maximum, idx_maximum = torch.max(similarity, dim=1, keepdim=True)
    idx_limit = similarity.size(1) - 1

    # Collect similarity scores for the disparities around the disparity
    # with the maximum similarity score.
    support_idx_disp = []
    for idx_shift in range(-half_support_window, half_support_window + 1):
        idx_disp = idx_maximum + idx_shift
        idx_disp[idx_disp < 0] = 0
        idx_disp[idx_disp >= idx_limit] = idx_limit
        support_idx_disp.append(idx_disp)

    support_idx_disp = torch.cat(support_idx_disp, dim=1)
    support_similar = torch.gather(similarity, 1, support_idx_disp.long())
    support_disp = support_idx_disp.float() * disp_step

    # Convert collected similarity scores to the disparity distribution
    # using softmax and compute disparity as a mean of this distribution.
    prob = F.softmax(support_similar, dim=1)
    disp = torch.sum(prob * support_disp.float(), dim=1)

    return disp

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_corrleation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, 2 * maxdisp + 1, H, W])
    for i in range(-maxdisp, maxdisp+1):
        if i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        elif i < 0:
            volume[:, :, i + maxdisp, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i],
                                                                     targetimg_fea[:, :, :, i:],
                                                                     num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def warp(x, disp):
    """
    warp an image/tensor (imright) back to imleft, according to the disp

    x: [B, C, H, W] (imright)
    disp: [B, 1, H, W] disp

    """
    B, C, H, W = x.size()
    device = x.get_device()
    # mesh grid
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    xx = xx.float()
    yy = yy.float()
    # grid = torch.cat((xx, yy), 1).float()

#     if x.is_cuda:
#         xx = xx.float().cuda()
#         yy = yy.float().cuda()
    xx_warp = Variable(xx) - disp
    yy = Variable(yy)
#     xx_warp = xx - disp
    vgrid = torch.cat((xx_warp, yy), 1)
    # vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size(), device=device, requires_grad=True)
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

def FMish(x):

    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''

    return x * torch.tanh(F.softplus(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   Mish())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
