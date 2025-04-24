import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from .update import BasicMultiUpdateBlock, ScaleBasicMultiUpdateBlock
from .extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock, DefomEncoder
from .corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from .utils.utils import coords_grid, upflow, get_danv2_io_size


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class DEFOMStereo(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.args = argparse.Namespace(
            mixed_precision=False,
            valid_iters=32,
            train_iters=18,
            scale_iters=8,
            dinov2_encoder='vits',
            idepth_scale=0.5,
            hidden_dims=[128]*3,
            corr_implementation='reg',
            shared_backbone=False,
            corr_levels=2,
            corr_radius=4,
            scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            scale_corr_radius=2,
            n_downsample=2,
            context_norm='batch',
            n_gru_layers=3,
        )

        if args is not None:
            for key, value in vars(args).items():
                setattr(self.args, key, value)

        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.defomencoder = DefomEncoder(self.args.dinov2_encoder, idepth_scale=self.args.idepth_scale)

        context_dims = self.args.hidden_dims

        self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=256, norm_fn='instance', downsample=self.args.n_downsample)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], self.args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=self.args.hidden_dims)
        self.scale_update_block = ScaleBasicMultiUpdateBlock(self.args, hidden_dims=self.args.hidden_dims)
        
        self.cnet = MultiBasicEncoder(self.defomencoder.out_dim, output_dim=[self.args.hidden_dims, context_dims],
                                      norm_fn=self.args.context_norm, downsample=self.args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_coords(self, img):
        """ Disparity is represented as difference between two vertical coordinate grids disp
            = coords0[:, :1] - coords1[:, :1] """
        N, _, H, W = img.shape

        coords = coords_grid(N, H, W)[:, :1].to(img.device)

        return coords

    def upsample_flow(self, flow, mask):
        """ Upsample disparity field [H/scale, W/scale, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=None, scale_iters=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        if iters is None:
            if self.training:
                iters = self.args.train_iters
            else:
                iters = self.args.valid_iters

        if scale_iters is None:
            scale_iters = self.args.scale_iters

        if self.training:
            self.args.mixed_precision = True
            test_mode = False
        else:
            self.args.mixed_precision = False
            test_mode = True

        # image1 = ((image1 - self.mean)/self.std).contiguous()
        # image2 = ((image2 - self.mean)/self.std).contiguous()

        bs, _, h, w = image1.shape
        danv2_io_sizes = get_danv2_io_size(h, w, self.args.n_downsample)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            d_features, dfeat1, dfeat2, disp = self.defomencoder([image1, image2], danv2_io_sizes)

            cnet_list = self.cnet(image1, d_features)
            fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        coords = self.initialize_coords(net_list[0])

        fmap1, fmap2 = fmap1.float(), fmap2.float()
        disp = disp.float()
        corr_fn = CorrBlock1D(fmap1, fmap2, coords, radius=self.args.corr_radius, num_levels=self.args.corr_levels,
                              scale_list=self.args.scale_list, scale_corr_radius=self.args.scale_corr_radius)

        disp_predictions = []
        for itr in range(iters):
            disp = disp.detach()

            if itr < scale_iters:
                corr = corr_fn(disp, scaling=True)  # index correlation volume
                with autocast(enabled=self.args.mixed_precision):
                    net_list, up_mask, scale_disp = self.scale_update_block(net_list, inp_list, corr, disp,
                                                                            iter32=self.args.n_gru_layers == 3,
                                                                            iter16=self.args.n_gru_layers >= 2)

                # F(t+1) = \Scale(t) x F(t)
                disp = scale_disp * disp
            else:
                corr = corr_fn(disp, scaling=False)  # index correlation volume
                with autocast(enabled=self.args.mixed_precision):
                    net_list, up_mask, delta_disp = self.update_block(net_list, inp_list, corr, disp,
                                                                      iter32=self.args.n_gru_layers == 3,
                                                                      iter16=self.args.n_gru_layers >= 2)

                    # To avoid unstability, we limit the disparity update within the searching range.
                    delta_disp = torch.clip(delta_disp, min=-2**(self.args.corr_levels-1)*self.args.corr_radius,
                                            max=2**(self.args.corr_levels-1)*self.args.corr_radius)

                # F(t+1) = F(t) + \Delta(t)
                disp = disp + delta_disp

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                disp_up = upflow(disp, factor=2 ** self.n_downsample)
            else:
                disp_up = self.upsample_flow(disp, up_mask)

            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up

        return disp_predictions
