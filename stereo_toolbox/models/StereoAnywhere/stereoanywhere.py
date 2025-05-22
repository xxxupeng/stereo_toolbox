import random
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

from .extractor import BasicEncoder, MultiBasicEncoder
from .update import BasicMultiUpdateBlock
from .corr import CorrBlock1D, CorrBlockFast1D
from .utils.utils import *
from .hourglass import Hourglass, HourglassIdentity
from .depth_anything_v2 import *

class StereoAnywhere(nn.Module):
    def __init__(self, args=SimpleNamespace()):
        super().__init__()

        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        self.args = args
        self.args.corr_implementation = "reg" if not hasattr(args, "corr_implementation") else args.corr_implementation
        self.args.n_downsample = 2 if not hasattr(args, "n_downsample") else args.n_downsample
        self.args.corr_radius = 4 if not hasattr(args, "corr_radius") else args.corr_radius
        self.args.corr_levels = 4 if not hasattr(args, "corr_levels") else args.corr_levels
        self.args.n_gru_layers = 3 if not hasattr(args, "n_gru_layers") else args.n_gru_layers
        self.args.encoder_output_dim = 128 if not hasattr(args, "encoder_output_dim") else args.encoder_output_dim
        self.args.context_dims = [128] * 3 if not hasattr(args, "context_dims") else args.context_dims

        self.args.n_additional_hourglass = 0 if not hasattr(args, "n_additional_hourglass") else args.n_additional_hourglass
        self.args.volume_channels = 8 if not hasattr(args, "volume_channels") else args.volume_channels
        self.args.vol_n_masks = 8 if not hasattr(args, "vol_n_masks") else args.vol_n_masks
        self.args.vol_aug_n_masks = 4 if not hasattr(args, "vol_aug_n_masks") else args.vol_aug_n_masks
        self.args.vol_downsample = 0 if not hasattr(args, "vol_downsample") else args.vol_downsample
        
        self.args.use_truncate_vol = True if not hasattr(args, "use_truncate_vol") else args.use_truncate_vol
        self.args.mirror_conf_th = 0.98 if not hasattr(args, "mirror_conf_th") else args.mirror_conf_th
        self.args.mirror_attenuation = 0.9 if not hasattr(args, "mirror_attenuation") else args.mirror_attenuation

        self.args.lrc_th = 1 if not hasattr(args, "lrc_th") else args.lrc_th
        self.args.moving_average_decay = 0.67 if not hasattr(args, "moving_average_decay") else args.moving_average_decay
        self.args.volume_corruption_prob = 0.33 if not hasattr(args, "volume_corruption_prob") else args.volume_corruption_prob
        self.args.normal_gain = 10 if not hasattr(args, "normal_gain") else args.normal_gain
        self.args.init_disparity_zero = False if not hasattr(args, "init_disparity_zero") else args.init_disparity_zero
        self.args.use_aggregate_stereo_vol = False if not hasattr(args, "use_aggregate_stereo_vol") else args.use_aggregate_stereo_vol
        self.args.use_aggregate_mono_vol = True if not hasattr(args, "use_aggregate_mono_vol") else args.use_aggregate_mono_vol
        self.args.things_to_freeze = ['fnet'] if not hasattr(args, "things_to_freeze") else args.things_to_freeze

        self.args.loadmonomodel = f'{Path(__file__).resolve().parent.parent}/depth_anything_v2/depth_anything_v2_vitl.pth' if not hasattr(args, "loadmonomodel") else args.loadmonomodel
        self.args.train_iters = 22 if not hasattr(args, "train_iters") else args.train_iters
        self.args.valid_iters = 32 if not hasattr(args, "valid_iters") else args.valid_iters

        self.cnet = MultiBasicEncoder(output_dim=[self.args.context_dims, self.args.context_dims], norm_fn="batch", downsample=self.args.n_downsample)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(self.args.context_dims[i], self.args.context_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=self.args.n_downsample)
        # self.light_fnet = FeatureV4(img_channels=1, n_downsample=self.args.n_downsample)

        self.feature_channels=[1, 1, 1, 1, 1, 1] # [64, 64, 64, 128, 192, 128]

        if self.args.use_aggregate_stereo_vol:
            self.hourglass_stereo = Hourglass(self.args.vol_n_masks, self.args.volume_channels, feature_channels=self.feature_channels, att_kernel_size=3, att_stride=1, att_padding=1)
            self.hourglass_stereo_stack = nn.ModuleList() 
            self.hourglass_stereo_stack.append(HourglassIdentity())
            for _ in range(self.args.n_additional_hourglass):
                self.hourglass_stereo_stack.append(Hourglass(self.args.volume_channels, self.args.volume_channels, feature_channels=self.feature_channels, att_kernel_size=3, att_stride=1, att_padding=1))
            self.classifier_stereo = nn.Conv3d(self.args.volume_channels, 1, 3, 1, 1, bias=False)

        self.hourglass_mono = Hourglass(self.args.vol_n_masks, self.args.volume_channels, feature_channels=self.feature_channels, att_kernel_size=3, att_stride=1, att_padding=1) # ex hourglass_stereo
        self.hourglass_mono_stack = nn.ModuleList() # ex hourglass_stereo2
        self.hourglass_mono_stack.append(HourglassIdentity())
        for _ in range(self.args.n_additional_hourglass):
            self.hourglass_mono_stack.append(Hourglass(self.args.volume_channels, self.args.volume_channels, feature_channels=self.feature_channels, att_kernel_size=3, att_stride=1, att_padding=1))
        self.classifier_mono = nn.Conv3d(self.args.volume_channels, 1, 3, 1, 1, bias=False) # ex classifier_stereo
        self.classifier_monoconf = nn.Conv3d(self.args.volume_channels, 1, 3, 1, 1, bias=False) # ex classifier_stereoconf

        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=self.args.context_dims, predict_confidence=False)

        self.mono_model = get_depth_anything_v2(self.args.loadmonomodel).eval()
        for param in self.mono_model.parameters():
            param.requires_grad = False
        for param in self.mono_model.buffers():
            param.requires_grad = False

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_for_finetuning(self):
        _freeze_dict = {
            'fnet': [self.fnet],
            'cnet': [self.cnet, self.context_zqr_convs],
            'monoagg': [self.hourglass_mono, self.hourglass_mono_stack, self.classifier_mono, self.classifier_monoconf],
        }

        for meta_layer in self.args.things_to_freeze:
            for layer in _freeze_dict[meta_layer]:
                for param in layer.parameters():
                    param.requires_grad = False
            
    def forward(self, image2, image3, iters=None, test_mode=False):
        if iters is None:
            if self.training:
                iters = self.args.train_iters
            else:
                iters = self.args.valid_iters

        if self.training:
            test_mode = False
        else:
            test_mode = True        

        B, C, H, W = image2.shape

        _input_size_width = 518 if abs(W - 518) <= abs(W - 518 *2) else 518*2
        _input_size_height = 518 if abs(H - 518) <= abs(H - 518 *2) else 518*2
        mono_depths = self.mono_model.infer_image(torch.cat([image2, image3], 0), 
                                                  input_size_width=_input_size_width, input_size_height=_input_size_height)
        mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
        mde2, mde3 = mono_depths[0].unsqueeze(0), mono_depths[1].unsqueeze(0)

        W_lowres = W // (2 ** self.args.n_downsample)

        if C == 1:
            image2 = torch.cat([image2 for _ in range(3)], 1)
            image3 = torch.cat([image3 for _ in range(3)], 1)
            image2, image3 = normalize([image2, image3])

        image2, image3 = image2 * 2 - 1, image3 * 2 - 1
        image2, image3 = image2.contiguous(), image3.contiguous()
 
        # mde2, mde3 = normalize([mde2, mde3])
        mde2, mde3 = mde2.contiguous(), mde3.contiguous()
        mde2_lowres = F.interpolate(mde2, scale_factor = 1 / (2 ** self.args.n_downsample), mode="bilinear", align_corners=True)
        mde3_lowres = F.interpolate(mde3, scale_factor = 1 / (2 ** self.args.n_downsample), mode="bilinear", align_corners=True)
        mde2_vollowres = F.interpolate(mde2, scale_factor = 1 / (2 ** self.args.vol_downsample), mode="bilinear", align_corners=True)
        mde3_vollowres = F.interpolate(mde3, scale_factor = 1 / (2 ** self.args.vol_downsample), mode="bilinear", align_corners=True)        
        mde2_normals_lowres = estimate_normals(mde2_lowres, normal_gain=(W_lowres / self.args.normal_gain))
        mde3_normals_lowres = estimate_normals(mde3_lowres, normal_gain=(W_lowres / self.args.normal_gain))
        
        cnet_list = self.cnet(torch.cat([mde2 for _ in range(3)], 1), num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        fmap2, fmap3 = self.fnet([image2, image3])
        #fmde2, fmde3 = self.light_fnet(mde2_vollowres), self.light_fnet(mde3_vollowres)
        fmde2 = [F.interpolate(mde2_vollowres, scale_factor=1/(2**i), mode="bilinear", align_corners=True) for i in range(self.args.n_downsample,len(self.feature_channels))]
        fmde3 = [F.interpolate(mde3_vollowres, scale_factor=1/(2**i), mode="bilinear", align_corners=True) for i in range(self.args.n_downsample,len(self.feature_channels))]
        fmap2, fmap3 = fmap2.float(), fmap3.float()
        
        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D        
        else:
            raise NotImplementedError()
                
        stereo_corr_volume = corr_block.corr(fmap2, fmap3).squeeze(3).unsqueeze(1)
        mono_corr_volume = 1.73 * corr_block.corr(mde2_normals_lowres, mde3_normals_lowres).squeeze(3).unsqueeze(1)
        
        left_volmasks_lowres = generate_masks(mde2_lowres, N=self.args.vol_n_masks)     #.unsqueeze(4) B N H/4 W/4 1
        right_volmasks_lowres = generate_masks(mde3_lowres, N=self.args.vol_n_masks)    #.unsqueeze(3) B N H/4  1 W/4
        
        if self.args.vol_downsample > 0:
            _original_shape = mono_corr_volume.shape # B 1 H W1 W2
            mono_corr_volume = F.interpolate(mono_corr_volume, scale_factor=1/(2**self.args.vol_downsample), mode="trilinear", align_corners=True)
            left_volmasks_lowres = F.interpolate(left_volmasks_lowres, scale_factor=1/(2**self.args.vol_downsample), mode="nearest")
            right_volmasks_lowres = F.interpolate(right_volmasks_lowres, scale_factor=1/(2**self.args.vol_downsample), mode="nearest")

        if self.args.use_aggregate_stereo_vol:
            masked_stereo_corr_volume = stereo_corr_volume * left_volmasks_lowres.unsqueeze(4) * right_volmasks_lowres.unsqueeze(3)
            agg_stereo_corr_volume = self.hourglass_stereo(masked_stereo_corr_volume, features_left=fmde2, features_right=fmde3) # B, C, H/4, W/4, W/4
            for i in range(self.args.n_additional_hourglass):
                agg_stereo_corr_volume = self.hourglass_stereo_stack[i](agg_stereo_corr_volume, features_left=fmde2, features_right=fmde3) # B, C, H/4, W/4, W/4
            agg_disp_stereo_corr = self.classifier_stereo(agg_stereo_corr_volume)

            coarse_dispstereo2_lowres = estimate_left_disparity(agg_disp_stereo_corr)
            coarse_dispstereo3_lowres = estimate_right_disparity(agg_disp_stereo_corr)
            coarse_dispstereo2 = F.interpolate(coarse_dispstereo2_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True) * (2**self.args.n_downsample)
            coarse_dispstereo3 = F.interpolate(coarse_dispstereo3_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True) * (2**self.args.n_downsample)
        else:
            coarse_dispstereo2, coarse_dispstereo3 = None, None

        masked_mono_corr_volume = mono_corr_volume * left_volmasks_lowres.unsqueeze(4) * right_volmasks_lowres.unsqueeze(3)
        agg_mono_corr_volume = self.hourglass_mono(masked_mono_corr_volume, features_left=fmde2, features_right=fmde3) # B, C, H/4, W/4, W/4
        for i in range(self.args.n_additional_hourglass):
            agg_mono_corr_volume = self.hourglass_mono_stack[i](agg_mono_corr_volume, features_left=fmde2, features_right=fmde3) # B, C, H/4, W/4, W/4
        agg_disp_mono_corr = self.classifier_mono(agg_mono_corr_volume)
        agg_conf_mono_corr = self.classifier_monoconf(agg_mono_corr_volume.detach())

        del masked_mono_corr_volume, agg_mono_corr_volume

        if self.args.vol_downsample > 0:         
            agg_disp_mono_corr = F.interpolate(agg_disp_mono_corr, (_original_shape[2], _original_shape[3], _original_shape[4]), mode="trilinear", align_corners=True)
            agg_conf_mono_corr = F.interpolate(agg_conf_mono_corr, (_original_shape[2], _original_shape[3], _original_shape[4]), mode="trilinear", align_corners=True)            
        
        coarse_dispmono2_lowres = estimate_left_disparity(agg_disp_mono_corr)
        coarse_dispmono3_lowres = estimate_right_disparity(agg_disp_mono_corr)
        coarse_ldispmonoconf2_lowres = estimate_left_confidence(agg_conf_mono_corr)
        coarse_ldispmonoconf3_lowres = estimate_right_confidence(agg_conf_mono_corr)

        del agg_conf_mono_corr

        coarse_dispmono2 = F.interpolate(coarse_dispmono2_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True) * (2**self.args.n_downsample)
        coarse_dispmono3 = F.interpolate(coarse_dispmono3_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True) * (2**self.args.n_downsample)
        coarse_ldispmonoconf2 = F.interpolate(coarse_ldispmonoconf2_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True)
        coarse_ldispmonoconf3 = F.interpolate(coarse_ldispmonoconf3_lowres, scale_factor=(2**self.args.n_downsample), mode="bilinear", align_corners=True)

        softlrc_coarse_dispmono2_lowres, softlrc_coarse_dispmono3_lowres = softlrc(coarse_dispmono2_lowres, coarse_dispmono3_lowres, lrc_th=self.args.lrc_th)

        coarse_dispmonoconf2_lowres = fuzzy_and(coarse_ldispmonoconf2_lowres, softlrc_coarse_dispmono2_lowres)
        coarse_dispmonoconf3_lowres = fuzzy_and(coarse_ldispmonoconf3_lowres, softlrc_coarse_dispmono3_lowres)

        global_scale_left, global_shift_left = weighted_lsq(torch.cat([mde2_lowres, mde3_lowres], 1), torch.cat([coarse_dispmono2_lowres, coarse_dispmono3_lowres], 1), torch.cat([coarse_dispmonoconf2_lowres, coarse_dispmonoconf3_lowres], 1))
        global_scale_right, global_shift_right = global_scale_left, global_shift_left

        coarse_scaled_mde2_lowres = torch.sum(global_scale_left * mde2_lowres + global_shift_left, dim=1, keepdim=True)
        coarse_scaled_mde2 = torch.sum(global_scale_left * mde2 + global_shift_left, dim=1, keepdim=True) * (2**self.args.n_downsample)
        coarse_scaled_mde3_lowres = torch.sum(global_scale_right * mde3_lowres + global_shift_right, dim=1, keepdim=True)
        coarse_scaled_mde3 = torch.sum(global_scale_right * mde3 + global_shift_right, dim=1, keepdim=True) * (2**self.args.n_downsample)

        softlrc_coarse_scaled_mde2_lowres, _ = softlrc(coarse_scaled_mde2_lowres, coarse_scaled_mde3_lowres, lrc_th=self.args.lrc_th)

        if self.args.use_truncate_vol:
            mde2_mirrorconf_lowres = handcrafted_mirror_detector(coarse_dispmono2_lowres, coarse_scaled_mde2_lowres, coarse_dispmonoconf2_lowres, softlrc_coarse_scaled_mde2_lowres, conf_th=self.args.mirror_conf_th)
            left_truncate_mask = truncate_corr_volume_v2(coarse_scaled_mde2_lowres, mde2_mirrorconf_lowres, conf_th=None, attenuation_gain=self.args.mirror_attenuation).detach()
        else:
            left_truncate_mask = 1

        #Vanilla raft-stereo corr volume
        _stereo_corr_volume = agg_disp_stereo_corr if self.args.use_aggregate_stereo_vol else stereo_corr_volume
        #Mono corr volume
        _mono_corr_volume = agg_disp_mono_corr if self.args.use_aggregate_mono_vol else mono_corr_volume

        coords0, coords1 = initialize_flow(net_list[0])

        if not test_mode:
            _vol_aug_masks_left = generate_masks(mde2_lowres, N=self.args.vol_aug_n_masks)     #.unsqueeze(4) B N H/4 W/4 1

        #Volume corruption augmentation: volume rolling (do not backpropagate)
        if random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            rolled_corr_volume = torch.roll(_stereo_corr_volume, shifts=random.randint(1, W_lowres), dims=3)
            _stereo_corr_volume = (_stereo_corr_volume * (1 - _left_mask) + rolled_corr_volume * _left_mask).detach()

        #Volume corruption augmentation: volume noising (do not backpropagate)
        elif random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            left_noise_curve = torch.rand_like(_left_mask)
            _stereo_corr_volume = (_stereo_corr_volume * (1 - _left_mask) + _stereo_corr_volume * left_noise_curve * _left_mask).detach()

        #Volume corruption augmentation: volume zeroing (do not backpropagate)
        elif random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            left_noise_curve = gauss_corr_volume_naive(torch.zeros_like(coarse_dispmono2_lowres), torch.max(_stereo_corr_volume).cpu().item())
            _stereo_corr_volume = (_stereo_corr_volume * (1 - _left_mask) + _stereo_corr_volume * left_noise_curve * _left_mask).detach()

        #MONO Volume corruption augmentation: volume rolling (do not backpropagate)
        elif random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            rolled_corr_volume = torch.roll(_mono_corr_volume, shifts=random.randint(1, W_lowres), dims=3)
            _mono_corr_volume = (_mono_corr_volume * (1 - _left_mask) + rolled_corr_volume * _left_mask).detach()

        #MONO Volume corruption augmentation: volume noising (do not backpropagate)
        elif random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            left_noise_curve = torch.rand_like(_left_mask)
            _mono_corr_volume = (_mono_corr_volume * (1 - _left_mask) + _mono_corr_volume * left_noise_curve * _left_mask).detach()

        #MONO Volume corruption augmentation: volume zeroing (do not backpropagate)
        elif random.random() < self.args.volume_corruption_prob and not test_mode:
            _left_mask = _vol_aug_masks_left[:, [random.randint(0, _vol_aug_masks_left.shape[1]-1)]].unsqueeze(4)
            left_noise_curve = gauss_corr_volume_naive(torch.zeros_like(coarse_dispmono2_lowres), torch.max(_mono_corr_volume).cpu().item())
            _mono_corr_volume = (_mono_corr_volume * (1 - _left_mask) + _mono_corr_volume * left_noise_curve * _left_mask).detach()

        stereo_corr_fn = corr_block(
            (left_truncate_mask*_stereo_corr_volume).squeeze(1).unsqueeze(3), radius=self.args.corr_radius, num_levels=self.args.corr_levels
        )

        mono_corr_fn = corr_block(
            (_mono_corr_volume).squeeze(1).unsqueeze(3), radius=self.args.corr_radius, num_levels=self.args.corr_levels
        )

        if not self.args.init_disparity_zero:
            coords1[:,:1] = coords0[:,:1] - coarse_scaled_mde2_lowres

        flow_predictions = []
        conf_predicitons = []

        for itr in range(iters):
            coords1 = coords1.detach()
            
            stereo_corr = stereo_corr_fn(coords1) # index correlation volume
            mono_corr = mono_corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            
            net_list, mask_up, delta_flow = self.update_block(net_list, inp_list, stereo_corr, mono_corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0            

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow 

            _flow_up = coords1 - coords0
            _flow_up = _flow_up[:,:1]

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # Upsample flow and confidence to full resolution
            flow_up = convex_upflow(_flow_up, mask_up, n_downsample=self.args.n_downsample, use_scale_factor=True)
            
            # Save predictions
            flow_predictions.append(-flow_up)
            conf_predicitons.append(None)

        if test_mode:
            return -flow_up
        
        return flow_predictions, conf_predicitons, [coarse_dispstereo2, coarse_dispmono2, coarse_scaled_mde2], [coarse_dispstereo3, coarse_dispmono3, coarse_scaled_mde3], [None, coarse_ldispmonoconf2, None], [None, coarse_ldispmonoconf3, None]
