import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import glob
import os.path as osp


def get_danv2_io_size(h, w, nds, max_i_size=2688, multiple_of=14): 
    """compute the input and output sizes of danv2 network"""
    danv2_oh, danv2_ow = h//2**nds, w//2**nds
    danv2_io_factor = 3.5  # more precise, 14/8=3.5
    ih, iw = danv2_io_factor*danv2_oh, danv2_io_factor*danv2_ow
    ih = int(np.ceil(ih / multiple_of) * multiple_of)
    iw = int(np.ceil(iw / multiple_of) * multiple_of)

    max_i_size = int(np.floor(max_i_size / multiple_of) * multiple_of)

    if ih <= max_i_size and iw <= max_i_size:
        danv2_ih, danv2_iw = ih, iw
    else:
        factor_h = max_i_size/ih
        factor_w = max_i_size/iw

        if factor_w > factor_h:
            danv2_ih = max_i_size
            danv2_iw = int(np.ceil(factor_h * iw / multiple_of) * multiple_of)
        else:
            danv2_iw = max_i_size
            danv2_ih = int(np.ceil(factor_w * ih / multiple_of) * multiple_of)

    return danv2_ih, danv2_iw, danv2_oh, danv2_ow
    

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    # img = bilinear_grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, factor=8, mode='bilinear', sacle=True):
    new_size = (factor * flow.shape[2], factor * flow.shape[3])
    if sacle:
        return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
    else:
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1, 1, N, N).to(input)
    output = F.conv2d(input.reshape(B*D, 1, H, W), weights, padding=N//2)
    return output.view(B, D, H, W)


# Ref: https://zenn.dev/pinto0309/scraps/7d4032067d0160
def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=im.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=im.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=im.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=im.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def read_kitti_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


# from https://github.com/ozendelait/rvc_devkit/blob/master/stereo/stereo_devkit.py
def ReadMiddlebury2014CalibFile(path):
    result = dict()
    with open(path, 'rb') as calib_file:
        for line in calib_file.readlines():
            line = line.decode('UTF-8').rstrip('\n')
            if len(line) == 0:
                continue
            eq_pos = line.find('=')
            if eq_pos < 0:
                raise Exception('Cannot parse Middlebury 2014 calib file: ' + path)
            result[line[:eq_pos]] = line[eq_pos + 1:]
    return result
