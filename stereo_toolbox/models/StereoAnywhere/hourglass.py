import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import BasicConv, DoubleFeatureAtt

class HourglassIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, features_left=None, features_right=None):
        return x

class Hourglass(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, norm_fn="instance", act_fn="lrelu", n_downsample=2, feature_channels=[64, 64, 64, 128, 192, 128], att_kernel_size=1, att_stride=1, att_padding=0):
        super(Hourglass, self).__init__()

        self.n_downsample = n_downsample
        self.feature_channels = feature_channels[self.n_downsample:]
        self.number_of_scales = len(self.feature_channels)

        # Initialize downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(self.number_of_scales-1):
            conv_in_channels = in_channels * (1  if i == 0 else 2 * i)
            conv_out_channels = in_channels * (2 * (i + 1))
            self.down_layers.append(
                nn.Sequential(
                    BasicConv(conv_in_channels, conv_out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=2, dilation=1, groups=groups),
                    BasicConv(conv_out_channels, conv_out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=1, dilation=1, groups=groups)
                )
            )

        # Initialize aggregation layers
        self.agg_layers = nn.ModuleList()
        for i in range(self.number_of_scales-2):
            agg_in_channels = in_channels * (2 * (self.number_of_scales - i - 1)) + in_channels * (2 * (self.number_of_scales - i - 2))
            agg_out_channels = in_channels * (2 * (self.number_of_scales - i - 2))
            self.agg_layers.append(
                nn.Sequential(
                    BasicConv(agg_in_channels, agg_out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=1, padding=0, stride=1),
                    BasicConv(agg_out_channels, agg_out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=1),
                    BasicConv(agg_out_channels, agg_out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=1)
                )
            )

        self.final_agg = nn.Sequential(
            BasicConv(in_channels + agg_out_channels, in_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels, in_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels, out_channels, is_3d=True, norm_fn=norm_fn, act_fn=act_fn, kernel_size=3, padding=1, stride=1)
        )

        self.feature_atts = nn.ModuleList([
            DoubleFeatureAtt(in_channels * (2 * i), self.feature_channels[i], kernel_size=att_kernel_size, stride=att_stride, padding=att_padding) for i in range(1, self.number_of_scales)
        ])
        self.feature_atts_up = nn.ModuleList([
            DoubleFeatureAtt(in_channels * (2 * (self.number_of_scales - i - 1)), self.feature_channels[self.number_of_scales - i - 1], kernel_size=att_kernel_size, stride=att_stride, padding=att_padding) for i in range(1, self.number_of_scales-1)
        ])

        self.final_feature_atts_up = DoubleFeatureAtt(out_channels, self.feature_channels[0], kernel_size=att_kernel_size, stride=att_stride, padding=att_padding)

    def forward(self, x, features_left, features_right):
        # Input shape permutation
        x = x.permute(0, 1, 3, 2, 4).permute(0, 1, 4, 3, 2)
        original_x = x

        # features_left = features_left[self.n_downsample:]
        # features_right = features_right[self.n_downsample:]

        # Downsample
        downsampled_features = []
        for i in range(self.number_of_scales-1):
            x = self.down_layers[i](x)
            x = self.feature_atts[i](x, features_left[i + 1], features_right[i + 1])
            downsampled_features.append(x)

        # Upsample and aggregate
        for i in range(self.number_of_scales - 2):
            _up_shape = downsampled_features[self.number_of_scales - 3 - i].shape
            x_up = F.interpolate(downsampled_features[self.number_of_scales - 2 - i], (_up_shape[2], _up_shape[3], _up_shape[4]), mode='trilinear', align_corners=True)
            x = torch.cat((x_up, downsampled_features[self.number_of_scales - 3 - i]), dim=1)
            x = self.agg_layers[i](x)
            x = self.feature_atts_up[i](x, features_left[self.number_of_scales - 2 - i], features_right[self.number_of_scales - 2 - i])

        # Final aggregation
        _up_shape = original_x.shape
        x_up = F.interpolate(x, (_up_shape[2], _up_shape[3], _up_shape[4]), mode='trilinear', align_corners=True)
        x = torch.cat((original_x, x_up), dim=1)
        x = self.final_agg(x)
        x = self.final_feature_atts_up(x, features_left[0], features_right[0])

        return x.permute(0, 1, 4, 3, 2).permute(0, 1, 3, 2, 4)
