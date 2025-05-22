import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import timm

from .dcn import DeformableConv2d

_ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'mish': nn.Mish,
    'none': nn.Identity
}

_NORMALIZATION_DICT = {
    'batch': nn.BatchNorm2d,
    'instance': nn.InstanceNorm2d,
    'batch3d': nn.BatchNorm3d,
    'instance3d': nn.InstanceNorm3d,
    'none': nn.Identity
}

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, norm_fn='instance', act_fn='lrelu', dcn=False, **kwargs):
        super(BasicConv, self).__init__()

        norm_fn = norm_fn.lower()+('3d' if is_3d else '')

        self.act_fn = _ACTIVATION_DICT.get(act_fn, nn.Identity)()
        self.norm_fn = _NORMALIZATION_DICT.get(norm_fn, nn.Identity)(out_channels)

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                if dcn:
                    self.conv = DeformableConv2d(in_channels, out_channels, bias=False, **kwargs)
                else:
                    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x

class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, norm_fn='instance', act_fn='lrelu', keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, norm_fn='instance', act_fn='lrelu', kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, norm_fn='instance', act_fn='lrelu', kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, norm_fn, act_fn, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, norm_fn, act_fn, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        #cv: B C W H W
        #feat: B C H W -> B C 1 H W
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

class DoubleFeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan_left, feat_chan_right = None, kernel_size=1, stride=0, padding=0):
        super(DoubleFeatureAtt, self).__init__()

        if feat_chan_right is None: 
            feat_chan_right = feat_chan_left

        self.feat_att_left = nn.Sequential(
            BasicConv(feat_chan_left, max(32,feat_chan_left//2), kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(max(32,feat_chan_left//2), cv_chan, 1))
        
        self.feat_att_right = nn.Sequential(
            BasicConv(feat_chan_right, max(32,feat_chan_right//2), kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(max(32,feat_chan_right//2), cv_chan, 1))

    def forward(self, cv, feat_left, feat_right):
        '''
        '''
        #cv: B C W H W
        #feat_left: B C H W -> B C 1 H W
        #feat_right: B C H W -> B C W H -> B C W H 1
        #feat_left*feat_right: B C W H W
        feat_att_left = self.feat_att_left(feat_left).unsqueeze(2)
        feat_att_right = self.feat_att_right(feat_right).permute(0,1,3,2).unsqueeze(4)
        _cv = torch.sigmoid(feat_att_left)*torch.sigmoid(feat_att_right)
        _cv = F.interpolate(_cv, size=(cv.shape[2], cv.shape[3], cv.shape[4]), mode='trilinear', align_corners=True)
        cv = _cv*cv
        return cv

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()

        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act1 # In the new version of timm, act1 is not present

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]

class FeatureV3(SubModule):
    def __init__(self):
        super(FeatureV4, self).__init__()

        self.model = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=True,
            features_only=True,
        )

        chans = [16, 16, 24, 48, 576]

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)              # 576 -> 48*2=96
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)             # 48*2 -> 24*2=48
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)              # 24*2 -> 16*2=32
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)   # 16*2 -> 16*2=32

    def forward(self, x):
        _,x4,x8,x16,x32 = self.model(x)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]

class FeatureV4(SubModule):
    def __init__(self, img_channels=3, n_downsample=2):
        super(FeatureV4, self).__init__()

        if n_downsample not in [0, 1, 2, 3]:
            raise ValueError('n_downsample must be in [0, 1, 2, 3] -- i.e, x1 x2 x4 x8 sub-sampling')

        self.img_channels = img_channels
        self.n_downsample = n_downsample
        self.feature_channels = [64, 64, 64, 128, 192, 128][self.n_downsample:]

        model = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            features_only=True,
        )

        self.mapping_conv = nn.Conv2d(img_channels, 3, kernel_size=1, stride=1, padding=0)

        chans = [32, 32, 64, 96, 128] # 2 4 8 16 32
        n_downsample_mapping = [0, 0, 1, 2]

        self.conv_stem = model.conv_stem # / 2
        self.bn1 = model.bn1
        self.act1 = model.act1 

        self.block0 = torch.nn.Sequential(*model.blocks[0:1])       # / 4 32
        self.block1 = torch.nn.Sequential(*model.blocks[1:2])       # / 8 64
        self.block2 = torch.nn.Sequential(*model.blocks[2:3])       # / 16 96
        self.block3 = torch.nn.Sequential(*model.blocks[3][0:1])    # / 32 128

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)              # 128 -> 96*2=192
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)             # 96*2 -> 64*2=128

        if n_downsample < 3:
            self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)              # 64*2 -> 32*2=64
        if n_downsample < 2:
            self.deconv4_2 = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)              # 32*2 -> 32*2=64    
                   
        _i = n_downsample_mapping[n_downsample]
        self.final_conv = BasicConv(chans[_i]*2, chans[_i]*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x2 = self.act1(self.bn1(self.conv_stem(self.mapping_conv(x))))
        x4 = self.block0(x2)
        x8 = self.block1(x4)
        x16 = self.block2(x8)
        x32 = self.block3(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)

        if self.n_downsample == 3:
            x8 = self.final_conv(x8)
            return [x8, x16, x32] # [128 192 128]
        if self.n_downsample == 2:
            x4 = self.deconv8_4(x8, x4)
            x4 = self.final_conv(x4)
            return [x4, x8, x16, x32] # [64 128 192 128]
        if self.n_downsample == 1:
            x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4, x2)
            x2 = self.final_conv(x2)
            return [x2, x4, x8, x16, x32] # [64 64 128 192 128]
        if self.n_downsample == 0:
            x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4, x2)
            x2 = self.final_conv(x2)
            x1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
            return [x1, x2, x4, x8, x16, x32] # [64 64 64 128 192 128]
