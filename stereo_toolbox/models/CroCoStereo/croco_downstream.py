# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# CroCo model for downstream tasks
# --------------------------------------------------------

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np

from .croco import CroCoNet
from .head_downstream import PixelwiseTaskWithDPT

def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt: # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()
    

def _get_gtnorm(gt):
    if gt.size(1)==1: # stereo
        return gt
    # flow 
    return torch.sqrt(torch.sum(gt**2, dim=1, keepdims=True)) # Bx1xHxW

def _resize_img(img, new_size):
    return F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)


def _resize_stereo_or_flow(data, new_size):
    assert data.ndim==4
    assert data.size(1) in [1,2]
    scale_x = new_size[1]/float(data.size(3))
    out = F.interpolate(data, size=new_size, mode='bicubic', align_corners=False)
    out[:,0,:,:] *= scale_x
    if out.size(1)==2:
        scale_y = new_size[0]/float(data.size(2))        
        out[:,1,:,:] *= scale_y
        print(scale_x, new_size, data.shape)
    return out


def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)


def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]


class LaplacianLossBounded2(nn.Module): # used for CroCo-Stereo (except for ETH3D) ; in the equation of the paper, we have a=b
    def __init__(self, max_gtnorm=None, a=3.0, b=3.0):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = True
        self.a, self.b = a, b
        
    def forward(self, predictions, gt, conf):
        mask = torch.isfinite(gt)
        mask = mask[:,0,:,:]
        if self.max_gtnorm is not None: mask *= _get_gtnorm(gt)[:,0,:,:]<self.max_gtnorm
        conf = conf.squeeze(1)
        conf = 2 * self.a * (torch.sigmoid(conf / self.b) - 0.5 )
        return ( torch.abs(gt-predictions).sum(dim=1)[mask] / torch.exp(conf[mask]) + conf[mask] ).mean()# + torch.log(2) => which is a constant


def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf


class CroCoDownstreamBinocular(CroCoNet):

    def __init__(self,
                 head=PixelwiseTaskWithDPT(),
                 enc_embed_dim = 1024,
                 enc_depth = 24, 
                 enc_num_heads = 16,
                 dec_embed_dim = 768,
                 dec_num_heads = 12,
                 dec_depth = 12,
                 img_size = (352, 704),
                 pos_embed = 'RoPE100',
                 conf_mode = 'conf_expsigmoid_15_3',
                 ):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDownstreamBinocular, self).__init__(
                 enc_embed_dim = enc_embed_dim,
                 enc_depth = enc_depth, 
                 enc_num_heads = enc_num_heads,
                 dec_embed_dim = dec_embed_dim,
                 dec_num_heads = dec_num_heads,
                 dec_depth = dec_depth,
                 img_size = img_size,
                 pos_embed = pos_embed,
                 )
        
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.dec_embed_dim = dec_embed_dim
        self.dec_num_heads = dec_num_heads
        self.dec_depth = dec_depth
        self.img_size = img_size
        self.conf_mode = conf_mode
        self.pos_embed = pos_embed

        num_channels = 1
        self.with_conf = eval('LaplacianLossBounded2(a=3, b=3)').with_conf
        if self.with_conf: num_channels += 1
        head.num_channels = num_channels
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head for downstream tasks, define your own head """
        return
        
    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        #out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        #out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        out, pos, _ = self._encode_image( torch.cat( (img1,img2), dim=0), do_mask=False, return_all_blocks=return_all_blocks )
        if return_all_blocks:
            out,out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))
            out2 = out2[-1]
        else:
            out,out2 = out.chunk(2, dim=0)
        pos,pos2 = pos.chunk(2, dim=0)            
        return out, out2, pos, pos2

    def forward_one_tile(self, img1, img2):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, out2, pos, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks)
        if return_all_blocks:
            decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
            decout = out+decout
        else:
            decout = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)
        return self.head(decout, img_info)
    
    @torch.no_grad()
    def forward(self, img1, img2, overlap=0.7):
        img1 = normalize_image(img1)
        img2 = normalize_image(img2)

        B, _, H, W = img1.shape
        C = self.head.num_channels-int(self.with_conf)

        win_height, win_width = self.img_size[0], self.img_size[1]
        do_change_scale =  H<win_height or W<win_width

        if do_change_scale: 
            upscale_factor = max(win_width/W, win_height/W)
            original_size = (H,W)
            new_size = (round(H*upscale_factor),round(W*upscale_factor))
            img1 = _resize_img(img1, new_size)
            img2 = _resize_img(img2, new_size)
            # resize gt just for the computation of tiled losses
            H,W = img1.shape[2:4]

        if self.conf_mode.startswith('conf_expsigmoid_'): # conf_expsigmoid_30_10
            beta, betasigmoid = map(float, self.conf_mode[len('conf_expsigmoid_'):].split('_'))
        elif self.conf_mode.startswith('conf_expbeta'): # conf_expbeta3
            beta = float(self.conf_mode[len('conf_expbeta'):])
        else:
            raise NotImplementedError(f"conf_mode {self.conf_mode} is not implemented")
        
        def crop_generator():
            for sy in _overlapping(H, win_height, overlap):
                for sx in _overlapping(W, win_width, overlap):
                    yield sy, sx, sy, sx, True

        # keep track of weighted sum of prediction*weights and weights
        accu_pred = img1.new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
        accu_conf = img1.new_zeros((B, H, W)) + 1e-16 # accumulate the weights 
        accu_c = img1.new_zeros((B, H, W)) # accumulate the weighted sum of confidences ; not so useful except for computing some losses

        for sy1, sx1, sy2, sx2, aligned in crop_generator():
            # compute optical flow there
            pred =  self.forward_one_tile(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))
            pred, predconf = split_prediction_conf(pred, with_conf=self.with_conf)
            
            if self.conf_mode.startswith('conf_expsigmoid_'):
                conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5)).view(B,win_height,win_width)
            elif self.conf_mode.startswith('conf_expbeta'):
                conf = torch.exp(- beta * predconf).view(B,win_height,win_width)
            else:
                raise NotImplementedError
                            
            accu_pred[...,sy1,sx1] += pred * conf[:,None,:,:]
            accu_conf[...,sy1,sx1] += conf
            accu_c[...,sy1,sx1] += predconf.view(B,win_height,win_width) * conf 
            
        pred = accu_pred / accu_conf[:, None,:,:]

        if do_change_scale:
            pred = _resize_stereo_or_flow(pred, original_size)

        return pred