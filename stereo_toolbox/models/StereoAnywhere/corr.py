import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    def __init__(self, fullcorr, num_levels=4, radius=4, pad=[0,0]):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.pad = pad

        # all pairs correlation
        self.fullcorr = fullcorr

        batch, h1, w1, dim, w2 = self.fullcorr.shape
        corr = self.fullcorr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, :1] # B 2 H W -> B 1 H W
        coords = coords + self.pad[0] # Real coords are shifted by pad[0]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            corr = corr.view(bz, -1, ht, wd)
            corr = corr[:, :, :, self.pad[0]:wd-self.pad[1]] # B 2r+1 H W' 
            out_pyramid.append()
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W1 = fmap2.shape
        _, _, _, W2 = fmap3.shape
        fmap2_dtype = fmap2.dtype

        fmap2 = fmap2.view(B, D, H, W1)
        fmap3 = fmap3.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return (corr / torch.sqrt(torch.tensor(D))).to(fmap2_dtype)

#Cannot create correlation volume dynamically
#class PytorchAlternateCorrBlock1D
#class AlternateCorrBlock

class CorrBlock1D:
    def __init__(self, fullcorr, num_levels=4, radius=4, pad = [0,0]):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.pad = pad

        # all pairs correlation
        self.fullcorr = fullcorr

        batch, h1, w1, dim, w2 = self.fullcorr.shape # B, H, W2, 1, W3
        corr = self.fullcorr.reshape(batch*h1*w1, dim, 1, w2) # BHW 1 W3

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1) # B 2 H W -> B 1 H W -> B H W 1
        coords = coords + self.pad[0] # Real coords are shifted by pad[0]
        batch, h1, w1, _ = coords.shape
        coords_dtype = coords.dtype

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device) # 1 1 2r+1 1
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i # BHW 1 2r+1 1
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1) # BHW 1 2r+1 2
            corr = bilinear_sampler(corr, coords_lvl) # BHW 1 2r+1
            corr = corr.view(batch, h1, w1, -1) # B H W 2r+1
            corr = corr[:, :, self.pad[0]:w1-self.pad[1], :] # B H W' 2r+1
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1) # B H W' (2r+1)*num_levels
        return out.permute(0, 3, 1, 2).contiguous().to(coords_dtype)# B (2r+1)*num_levels H W'

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W2 = fmap2.shape
        _, _, _, W3 = fmap3.shape
        fmap2_dtype = fmap2.dtype

        fmap2 = fmap2.view(B, D, H, W2)
        fmap3 = fmap3.view(B, D, H, W3)

        # a i j k: batch, feature, height, width
        # a i j h: batch, feature, height, disparity
        # a j k h: batch, height, width, disparity

        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W2, 1, W3).contiguous()
        return (corr / torch.sqrt(torch.tensor(D))).to(fmap2_dtype)
