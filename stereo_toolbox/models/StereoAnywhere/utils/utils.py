import torch
import torch.nn.functional as F
from kornia.filters import spatial_gradient
import math
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

def colormap_image(x,vmax=None,cmap='Spectral_r'):
    """Apply colormap to image pixels
    """
    ma = float(x.max()) if vmax is None else vmax
    mi = float(0)
    normalizer = mpl.colors.Normalize(vmin=mi, vmax=ma)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = mapper.to_rgba(x)[:, :, :3]
    return colormapped_im

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    img_dtype = img.dtype

    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img.float(), grid, align_corners=True).to(img_dtype)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.to(img_dtype)

    return img

def coords_grid(batch, ht, wd, dtype, device):
    coords = torch.meshgrid(torch.arange(ht, dtype=dtype, device=device), torch.arange(wd, dtype=dtype, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).to(dtype).to(device)
    return coords[None].repeat(batch, 1, 1, 1).to(device) # B 2 H W

def upflow(flow, factor=2, mode='bilinear', use_scale_factor = True):
    scale = 2 ** factor
    new_size = (scale * flow.shape[2], scale * flow.shape[3])
    _tmp = F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
    return _tmp * scale if use_scale_factor else _tmp

def generate_masks(mde, N=16):
    B, _, H, W = mde.shape
    masks = torch.zeros(B, N, H, W, dtype=torch.float16, device=mde.device)
    for i in range(N):
        mask = (mde < (i+1)/N) & (mde >= i/N)
        masks[:, i] = mask.squeeze(1)
    return masks # B N H W

def normalize(x, eps=1e-4):
    if not isinstance(x, list):
        x = [x]

    prev_min = None
    prev_max = None
        
    for i in range(len(x)):
        _min = -F.max_pool2d(-x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _min = _min if prev_min is None else torch.min(_min, prev_min)
        _max = F.max_pool2d(x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _max = _max if prev_max is None else torch.max(_max, prev_max)
        prev_min = _min
        prev_max = _max
        
    return [(_x-_min)/(_max-_min+eps) for _x in x]

def estimate_normals(depth, normal_gain):
    xy_gradients = -spatial_gradient(normal_gain*depth, mode='diff', order=1, normalized=False).squeeze(1) # B 2 H W
    normals = torch.cat([xy_gradients, torch.ones_like(xy_gradients[:,0:1])], 1) # B 3 H W
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    return normals

def estimate_gradient_magnitude(depth, gradient_gain):
    xy_gradients = spatial_gradient(depth*gradient_gain, mode='diff', order=1, normalized=False).squeeze(1) # B 2 H W
    magnitude = torch.linalg.norm(xy_gradients, dim=1, keepdim=True)
    return magnitude

def edge_confidence(depth, gradient_gain, edge_gain):
    grad_magnitude = estimate_gradient_magnitude(depth, gradient_gain)
    return 1-torch.exp(-edge_gain*grad_magnitude)

def initialize_flow(img):
    """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
    N, _, H, W = img.shape

    coords0 = coords_grid(N, H, W, img.dtype, img.device)
    coords1 = coords_grid(N, H, W, img.dtype, img.device)

    return coords0, coords1
    
def convex_upflow(flow, mask, n_downsample=2, use_scale_factor = True):
        """ Upsample flow field [H/F, W/F, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        _tmp = factor * flow if use_scale_factor else flow
        up_flow = F.unfold(_tmp, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

def estimate_left_disparity(corr_volume, vol_pad=[0,0]):
    B, _, H, W2, W3 = corr_volume.shape # B 1 H W W

    disp_values = torch.arange(0, W3, dtype=corr_volume.dtype, device=corr_volume.device)
    disp_values_left = disp_values.view(1, 1, 1, -1).repeat(B, 1, 1, 1) # B 1 1 W3

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=corr_volume.dtype, device=corr_volume.device), torch.arange(W2, dtype=corr_volume.dtype, device=corr_volume.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(corr_volume.device) # B H W2
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(corr_volume.device) # B H W2

    #prob masking
    # mask = (mycoords_x.unsqueeze(3) >= disp_values_left).float()
    mask = 1.0

    prob_left = F.softmax(corr_volume.squeeze(1) * mask, dim=3)# B, 1, H, W2, W3 -> B, H, W2, W3
    prob_left = torch.sum(prob_left * disp_values_left, 3, keepdim=False) # B H W2

    disp_left = (mycoords_x-prob_left) # B H W2
    disp_left = disp_left.unsqueeze(1) # B (1) H W2
    return disp_left[:, :, :, vol_pad[0]:W2-vol_pad[1]]

def estimate_right_disparity(corr_volume, vol_pad=[0,0]):
    B, _, H, W2, W3 = corr_volume.shape

    disp_values = torch.arange(0, W2, dtype=corr_volume.dtype, device=corr_volume.device)
    disp_values_right = disp_values.view(1, 1, -1, 1).repeat(B, 1, 1, 1) # B 1 W2 1

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=corr_volume.dtype, device=corr_volume.device), torch.arange(W3, dtype=corr_volume.dtype, device=corr_volume.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(corr_volume.device) # B H W3
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(corr_volume.device) # B H W3

    #prob masking
    # mask = (mycoords_x.unsqueeze(2) <= disp_values_right).float()
    mask = 1.0

    prob_right = F.softmax(corr_volume.squeeze(1) * mask, dim=2)# B, 1, H, W2, W3 -> B, H, W2, W3
    prob_right = torch.sum(prob_right * disp_values_right, 2, keepdim=False) # B H W3
    
    disp_right = (prob_right-mycoords_x) # B H W3
    disp_right = disp_right.unsqueeze(1) # B (1) H W3
    return disp_right[:, :, :, vol_pad[0]:W3-vol_pad[1]]

def estimate_left_confidence(corr_volume, logsumexp_eps=1e-3):
    _, _, _, _, W3 = corr_volume.shape
    prob_left = F.softmax(corr_volume.squeeze(1), dim=3) # B, 1, H, W2, W3 -> B, H, W2, W3
    # conf_left = logsumexp_eps * torch.logsumexp(prob_left/logsumexp_eps, dim=3, keepdim=False) # B H W2
    #Alternative based on information entropy
    conf_left = -torch.sum(prob_left * torch.log2(prob_left+1e-6), dim=3, keepdim=False) / math.log2(W3) # B H W2
    conf_left = 1 - conf_left # High confidence for low entropy
    return conf_left.unsqueeze(1) # B 1 H W2

def estimate_right_confidence(corr_volume, logsumexp_eps=1e-3):
    _, _, _, W2, _ = corr_volume.shape
    prob_right = F.softmax(corr_volume.squeeze(1), dim=2) # B, 1, H, W2, W3 -> B, H, W2, W3
    # conf_right = logsumexp_eps * torch.logsumexp(prob_right/logsumexp_eps, dim=2, keepdim=False) # B H W3
    #Alternative based on information entropy
    conf_right = -torch.sum(prob_right * torch.log2(prob_right+1e-6), dim=2, keepdim=False) / math.log2(W2) # B H W3
    conf_right = 1 - conf_right # High confidence for low entropy
    return conf_right.unsqueeze(1) # B 1 H W3

def disp_warping(disp, img, right_disp=False):
    B, _, H, W = disp.shape

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=disp.dtype, device=disp.device), torch.arange(W, dtype=disp.dtype, device=disp.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp.device)
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp.device)

    if right_disp:
        grid = 2 * torch.cat([(mycoords_x+disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1
    else:
        grid = 2 * torch.cat([(mycoords_x-disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1

    # grid_sample: B,C,H,W & B H W 2 -> B C H W
    warped_img = F.grid_sample(img, grid, align_corners=True)

    return warped_img

def softlrc(disp2, disp3, lrc_th=1.0):
    div_const = math.log(1+math.exp(lrc_th))

    warped_disp2 = disp_warping(F.relu(disp3), disp2, right_disp=True) # B 1 H W
    warped_disp3 = disp_warping(F.relu(disp2), disp3, right_disp=False) # B 1 H W

    softlrc_disp2 = F.softplus(-torch.abs(disp2-warped_disp3)+lrc_th) / div_const # lrc weights in (0,1)  #B 1 H W
    softlrc_disp3 = F.softplus(-torch.abs(disp3-warped_disp2)+lrc_th) / div_const # lrc weights in (0,1) #B 1 H W  

    return softlrc_disp2, softlrc_disp3

def gauss_corr_volume_naive(disp_left, gauss_k = 10, gauss_c = 1):
    B, _, H, W = disp_left.shape

    disp_values = torch.arange(0, W, dtype=disp_left.dtype, device=disp_left.device)
    disp_values_left = disp_values.view(1, 1, 1, -1).repeat(B, 1, 1, 1) # B 1 1 W/4
    
    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=disp_left.dtype, device=disp_left.device), torch.arange(W, dtype=disp_left.dtype, device=disp_left.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp_left.device)
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp_left.device)

    gauss_center = (mycoords_x.unsqueeze(1).unsqueeze(4)-disp_left.unsqueeze(4)) #B 1 H/4 W/4 1
    gauss_corr_left = (gauss_center-disp_values_left.unsqueeze(1)) #B 1 H/4 W/4 W/4
    gauss_corr_left = gauss_k * torch.exp(-(gauss_corr_left**2)/(2*gauss_c**2)) #B 1 H/4 W/4 W/4
    
    return gauss_corr_left

def truncate_corr_volume_v2(disp_left, conf_left, conf_th = 0.5, attenuation_gain = 0.1):
    B, _, H, W = disp_left.shape
    disp_left_dtype = disp_left.dtype

    disp_values = torch.arange(0, W, dtype=disp_left.dtype, device=disp_left.device)
    disp_values_left = disp_values.view(1, 1, 1, -1).repeat(B, 1, 1, 1) # B 1 1 W/4

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=disp_left.dtype, device=disp_left.device), torch.arange(W, dtype=disp_left.dtype, device=disp_left.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp_left.device)
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp_left.device)

    if conf_th is not None:
        conf_left = (conf_left > conf_th).to(disp_left_dtype)
    conf_left = conf_left.unsqueeze(4)       # B 1 H/4 W/4 1

    truncate_center = (mycoords_x.unsqueeze(1).unsqueeze(4)-disp_left.unsqueeze(4)) #B 1 H/4 W/4 1
    truncate_corr_left = (truncate_center-disp_values_left.unsqueeze(1)) #B 1 H/4 W/4 W/4

    # I'm expecting the pixel corrispondece to be in the left side of truncate_center
    # Sigmoid sign tested with a simple example
    truncate_corr_left = 1 * (1-conf_left) + (conf_left) * (F.sigmoid(truncate_corr_left) * (1-attenuation_gain) + attenuation_gain)

    return truncate_corr_left

def fuzzy_and(x, y):
    return x*y

def fuzzy_or(x, y):
    return x+y-x*y

def fuzzy_not(x):
    return 1-x

def fuzzy_and_zadeh(x, y, eps=1e-3):
    return -eps*torch.logsumexp(-torch.cat([x, y], 1) / eps, 1, keepdim=True)

def fuzzy_or_zadeh(x, y, eps=1e-3):
    return eps*torch.logsumexp(torch.cat([x, y], 1) / eps, 1, keepdim=True)

def handcrafted_mirror_detector(stereo_disp, mono_disp, stereo_conf, mono_conf, conf_th=0.5, step_gain=20):
    # Handcrafted confidence: (MONO >> STEREO AND LRC_MONO) OR (LRC_MONO AND ~LRC_STEREO)
    # Four cases:
    # BOTH LRCs are bad (LRC_MONO=0; LRC_STEREO=0): we are in occlusions where mono is typically better than stereo, but if scale is wrong mono is not trustable
    # LRC_STEREO is bad (LRC_MONO=1; LRC_STEREO=0): probably there are high-frequency details better captured by mono
    # LRC_MONO is bad (LRC_MONO=0; LRC_STEREO=1): probably there is an optical illusion in the stereo pair or mono is not consistent with the stereo pair
    # BOTH LRCs are good (LRC_MONO=1; LRC_STEREO=1): probably the stereo pair is consistent with the mono prediction: usually stereo is better here, however, mono is predicting high disparity values probably there is a mirror

    mono_and_stereo_conf = fuzzy_and(stereo_conf, mono_conf)
    mono_near_wrt_stereo = F.sigmoid(step_gain * (mono_disp - stereo_disp))
    mono_is_better_a = fuzzy_and(mono_and_stereo_conf, mono_near_wrt_stereo)
    mono_is_better_b = fuzzy_and(fuzzy_not(stereo_conf), mono_conf)
    mono_is_better = fuzzy_or(mono_is_better_a, mono_is_better_b)

    return F.sigmoid(step_gain * (mono_is_better-conf_th))

def corr(normals_left, normals_right):
    B, D, H, W2 = normals_left.shape
    _, _, _, W3 = normals_right.shape
    normals_left_dtype = normals_left.dtype

    normals_left = normals_left.view(B, D, H, W2)
    normals_right = normals_right.view(B, D, H, W3)

    corr = torch.einsum('aijk,aijh->ajkh', normals_left, normals_right)
    corr = corr.reshape(B, H, W2, 1, W3).contiguous()
    corr = (corr / torch.sqrt(torch.tensor(D))).to(normals_left_dtype)
    
    return corr.squeeze(3).unsqueeze(1) #torch.Size([B, H, W1, 1, W2]) -> B, 1, H, W1, W2

def correlation_score(normals_a, normals_b):
    B, C, H, W = normals_a.shape

    normals_a = normals_a.view(B, C, H, W)
    normals_b = normals_b.view(B, C, H, W)

    corr_score = torch.sum(normals_a * normals_b, dim=1, keepdim=True) # B 1 H W
    
    return corr_score

def normalized_depth_scale_and_shift(
    prediction, target, mask, min_quantile = 0.2, max_quantile = 0.9
):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """

    B, _, _, _ = prediction.shape
    
    if min_quantile > 0.0 or max_quantile < 1.0:
        # compute quantiles
        target_dtype = target.dtype
        min_quantile = torch.quantile(target.float(), min_quantile).to(target_dtype)
        max_quantile = torch.quantile(target.float(), max_quantile).to(target_dtype)
        mask = (target >= min_quantile) * (target <= max_quantile) * mask

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3))
    a_01 = torch.sum(mask * prediction, (1, 2, 3))
    a_11 = torch.sum(mask, (1, 2, 3))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3))
    b_1 = torch.sum(mask * target, (1, 2, 3))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = (det != 0)

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale.reshape(B,1,1,1), shift.reshape(B,1,1,1)

def weighted_lsq(mde, disp, conf, min_quantile = 0.2, max_quantile = 0.9):
    B, _, _, _ = mde.shape
    mde_dtype = mde.dtype

    # Weighted LSQ
    mde, disp, conf = mde.reshape(B, -1).float(), disp.reshape(B, -1).float(), conf.reshape(B, -1).float()

    disp = F.relu(disp)

    scale_shift = torch.zeros((B, 2), device=mde.device)

    for b in range(B):
        _mono = mde[b].unsqueeze(0)
        _stereo = disp[b].unsqueeze(0)
        _conf = conf[b].unsqueeze(0)

        _min_disp = torch.quantile(_stereo.flatten(), min_quantile)
        _max_disp = torch.quantile(_stereo.flatten(), max_quantile)

        _quantile_mask = (_min_disp <= _stereo) & (_stereo <= _max_disp)

        _mono = _mono[_quantile_mask].unsqueeze(0)
        _conf = _conf[_quantile_mask].unsqueeze(0)
        _stereo = _stereo[_quantile_mask].unsqueeze(0)

        _mono = torch.abs(_mono.flatten().unsqueeze(0))
        _stereo = torch.abs(_stereo.flatten().unsqueeze(0))
        _conf = torch.abs(_conf.flatten().unsqueeze(0))

        _conf = _conf * (1-0.1) + 0.1
        
        weights = torch.sqrt(_conf)
        A_matrix = _mono * weights
        A_matrix = torch.cat([A_matrix.unsqueeze(-1), weights.unsqueeze(-1)], -1)
        B_matrix = (_stereo * weights).unsqueeze(-1)

        _scale_shift = torch.linalg.lstsq(A_matrix, B_matrix)[0].squeeze(2) # 1 x 2 x 1 -> 1 x 2,
        scale_shift[b] = _scale_shift.squeeze(0)

    return scale_shift[:, 0:1].reshape(B,1,1,1).to(mde_dtype), scale_shift[:, 1:2].reshape(B,1,1,1).to(mde_dtype)
    
def naive_scale_shift(mde, disp, conf, conf_th = 0.5):
    B, _, _, _ = mde.shape

    scale_and_shift_values = torch.zeros((B, 2), device=mde.device)

    for b in range(B):
        _mde = mde[b].unsqueeze(0)
        _disp = disp[b].unsqueeze(0)
        _conf = conf[b].unsqueeze(0)

        _mde = _mde[_conf > conf_th].unsqueeze(0)
        _disp = _disp[_conf > conf_th].unsqueeze(0)

        mde_90_percentile = torch.quantile(_mde, 0.9)
        mde_median = torch.median(_mde)

        disp_90_percentile = torch.quantile(_disp, 0.9)
        disp_median = torch.median(_disp)

        scale = (disp_90_percentile - disp_median) / (mde_90_percentile - mde_median)
        shift = disp_median - scale * mde_median

        scale_and_shift_values[b] = torch.tensor([scale, shift], device=mde.device)
    
    return scale_and_shift_values[:, 0:1].reshape(B,1,1,1), scale_and_shift_values[:, 1:2].reshape(B,1,1,1)

def apply_scale_shift(mde, scales, shifts, masks=None):
    if masks is None:
        masks = torch.ones_like(mde)
    
    B, N, _, _ = masks.shape

    mde = mde.repeat(1, N, 1, 1)
    scaled_mdes = scales * mde + shifts
    scaled_mdes = scaled_mdes * masks
    
    return torch.sum(scaled_mdes, dim=1, keepdim=True)