from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

from .submodule import *


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer5 = self._make_layer(BasicBlock, 192, 3, 2, 1, 1)
        # self.layer6 = self._make_layer(BasicBlock, 192, 3, 1, 1, 1)
        self.layer7 = self._make_layer(BasicBlock, 256, 3, 2, 1, 1)
        # self.layer8 = self._make_layer(BasicBlock, 256, 3, 1, 1, 1)
        self.layer9 = self._make_layer(BasicBlock, 512, 3, 2, 1, 1)
        # self.layer10 = self._make_layer(BasicBlock, 512, 3, 1, 1, 1)

        self.gw2 = nn.Sequential(convbn(192, 320, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.gw3 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.layer11 = nn.Sequential(convbn(320, 320, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
        self.layer_refine = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          Mish(),
                                          convbn(128, 32, 1, 1, 0, 1),
                                          Mish())
        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat2 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
            self.concat3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat4 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))





    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)    #1/4
        l5 = self.layer5(l4)    #1/8
        # l6 = self.layer6(l5)
        l6 = self.layer7(l5)    #1/16
        # l8 = self.layer8(l7)
        l7 = self.layer9(l6)    #1/32
        # l10 = self.layer10(l9)


        featurecombine = torch.cat((l2, l3, l4), dim=1)
        # combine1 = torch.cat((l5, l6), dim=1)
        # combine2 = torch.cat((l7, l8), dim=1)
        # combine3 = torch.cat((l9, l10), dim=1)
        gw1 = self.layer11(featurecombine)
        gw2 = self.gw2(l5)
        gw3 = self.gw3(l6)
        gw4 = self.gw4(l7)
        feature_refine = self.layer_refine(featurecombine)

        if not self.concat_feature:
            return {"gw1": gw1, "gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            concat_feature1 = self.lastconv(featurecombine)
            concat_feature2 = self.concat2(l5)
            concat_feature3 = self.concat3(l6)
            concat_feature4 = self.concat4(l7)
            return {"gw1": gw1, "gw2": gw2, "gw3": gw3, "gw4": gw4,  "concat_feature1": concat_feature1, "finetune_feature": feature_refine,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3, "concat_feature4": concat_feature4}

class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        # self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
        #                            Mish())
        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        # self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        # self.conv5 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv5 = nn.Conv3d(in_channels * 4, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv6 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())
        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 4, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 4))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x, feature4, feature5, feature6):
        conv1 = self.conv1(x)          #1/8
        conv1 = torch.cat((conv1, feature4), dim=1)   #1/8
        conv1 = self.combine1(conv1)   #1/8
        conv2 = self.conv2(conv1)      #1/8

        conv3 = self.conv3(conv2)      #1/16
        conv3 = torch.cat((conv3, feature5), dim=1)   #1/16
        conv3 = self.combine2(conv3)   #1/16
        conv4 = self.conv4(conv3)      #1/16

        conv5 = self.conv5(conv4)      #1/32
        conv5 = torch.cat((conv5, feature6), dim=1)  #1/32
        conv5 = self.combine3(conv5)
        conv6 = self.conv6(conv5)      #1/32

        # conv7 = F.relu(self.conv7(conv6) + self.redir3(conv4), inplace=True)
        # conv8 = F.relu(self.conv8(conv7) + self.redir2(conv2), inplace=True)
        # conv9 = F.relu(self.conv9(conv8) + self.redir1(x), inplace=True)

        conv7 = FMish((self.conv7(conv6) + self.redir3(conv4)))
        conv8 = FMish(self.conv8(conv7) + self.redir2(conv2))
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))


        return conv9

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6


class refinenet_version3(nn.Module):
    def __init__(self, in_channels):
        super(refinenet_version3, self).__init__()

        self.inplanes = 128
        self.conv1 = nn.Sequential(
            convbn(in_channels, 128, 3, 1, 1, 1),
            Mish())

        # self.conv2 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        # self.conv3 = self._make_layer(BasicBlock, 128, 1, 1, 1, 2)
        # self.conv4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 4)
        self.conv2 = nn.Sequential(
            convbn(128, 128, 3, 1, 1, 1),
            Mish())
        self.conv3 = nn.Sequential(
            convbn(128, 128, 3, 1, 2, 2),
            Mish())
        self.conv4 = nn.Sequential(
            convbn(128, 128, 3, 1, 4, 4),
            Mish())
        self.conv5 = self._make_layer(BasicBlock, 96, 1, 1, 1, 8)
        self.conv6 = self._make_layer(BasicBlock, 64, 1, 1, 1, 16)
        self.conv7 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)

        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x, disp):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        disp = disp + conv8

        return disp

class PCWNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(PCWNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.combine1 = hourglassup(32)

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.refinenet3 = refinenet_version3(146)
        self.dispupsample = nn.Sequential(convbn(1, 32, 1, 1, 0, 1),
                                          Mish())

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
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume1 = build_gwc_volume(features_left["gw1"], features_right["gw1"], self.maxdisp // 4,
                                      self.num_groups)

        gwc_volume2 = build_gwc_volume(features_left["gw2"], features_right["gw2"], self.maxdisp // 8,
                                       self.num_groups)

        gwc_volume3 = build_gwc_volume(features_left["gw3"], features_right["gw3"], self.maxdisp // 16,
                                       self.num_groups)

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 32,
                                       self.num_groups)
        if self.use_concat_volume:
            concat_volume1 = build_concat_volume(features_left["concat_feature1"], features_right["concat_feature1"],
                                                self.maxdisp // 4)
            concat_volume2 = build_concat_volume(features_left["concat_feature2"], features_right["concat_feature2"],
                                                 self.maxdisp // 8)
            concat_volume3 = build_concat_volume(features_left["concat_feature3"], features_right["concat_feature3"],
                                                 self.maxdisp // 16)
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"],
                                                 self.maxdisp // 32)
            volume1 = torch.cat((gwc_volume1, concat_volume1), 1)
            volume2 = torch.cat((gwc_volume2, concat_volume2), 1)
            volume3 = torch.cat((gwc_volume3, concat_volume3), 1)
            volume4 = torch.cat((gwc_volume4, concat_volume4), 1)
        else:
            volume1 = gwc_volume1
            volume2 = gwc_volume2
            volume3 = gwc_volume3
            volume4 = gwc_volume4

        cost0 = self.dres0(volume1)
        cost0 = self.dres1(cost0) + cost0

        combine = self.combine1(cost0, volume2, volume3, volume4)
        # combine = self.combine1(cost0)
        out1 = self.dres2(combine)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            # refinement = self.dia6(self.dia5(self.dia4(self.dia3(self.dia2(self.dia1(out3))))))
            # cost3 = cost3 + refinement
            combine = self.classif4(combine)
            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            combine = F.upsample(combine, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            combine = torch.squeeze(combine, 1)
            combine = F.softmax(combine, dim=1)
            combine = disparity_regression(combine, self.maxdisp)

            pred3 = torch.unsqueeze(pred3, 1)
            refinenet_feature_left = features_left["finetune_feature"]
            refinenet_feature_left = F.upsample(refinenet_feature_left, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            refinenet_feature_right = features_right["finetune_feature"]
            refinenet_feature_right = F.upsample(refinenet_feature_right, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            refinenet_feature_right_warp = warp(refinenet_feature_right, pred3)
            refinenet_costvolume = build_corrleation_volume(refinenet_feature_left, refinenet_feature_right_warp, 24, 1)
            refinenet_costvolume = torch.squeeze(refinenet_costvolume, 1)
            pred3feature = self.dispupsample(pred3)
            refinenet_combine = torch.cat((refinenet_feature_left - refinenet_feature_right_warp, refinenet_feature_left, pred3feature, pred3, refinenet_costvolume), dim = 1)
            disp_finetune = self.refinenet3(refinenet_combine, pred3)
            disp_finetune = torch.squeeze(disp_finetune, 1)
            pred3 = torch.squeeze(pred3, 1)
            # pred3 = torch.unsqueeze(pred3, 1)
            # pred3feature = self.dispupsample(pred3)
            # refinenet_combine = torch.cat((refinenet_feature, pred3feature, left), dim = 1)
            # disp_finetune = self.refinenet3(refinenet_combine, pred3)
            # disp_finetune = torch.squeeze(disp_finetune, 1)
            # pred3 = torch.squeeze(pred3, 1)

            return [pred0, combine, pred1, pred2, pred3, disp_finetune]

        else:
            cost3 = self.classif3(out3)
            # refinement = self.dia6(self.dia5(self.dia4(self.dia3(self.dia2(self.dia1(out3))))))
            # cost3 = cost3 + refinement
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            pred3 = torch.unsqueeze(pred3, 1)
            refinenet_feature_left = features_left["finetune_feature"]
            refinenet_feature_left = F.upsample(refinenet_feature_left, [left.size()[2], left.size()[3]],
                                                mode='bilinear', align_corners=True)
            refinenet_feature_right = features_right["finetune_feature"]
            refinenet_feature_right = F.upsample(refinenet_feature_right, [left.size()[2], left.size()[3]],
                                                 mode='bilinear', align_corners=True)
            refinenet_feature_right_warp = warp(refinenet_feature_right, pred3)
            refinenet_costvolume = build_corrleation_volume(refinenet_feature_left, refinenet_feature_right_warp, 24, 1)
            refinenet_costvolume = torch.squeeze(refinenet_costvolume, 1)
            pred3feature = self.dispupsample(pred3)
            refinenet_combine = torch.cat((
                                          refinenet_feature_left - refinenet_feature_right_warp, refinenet_feature_left,
                                          pred3feature, pred3, refinenet_costvolume), dim=1)
            disp_finetune = self.refinenet3(refinenet_combine, pred3)
            disp_finetune = torch.squeeze(disp_finetune, 1)
            pred3 = torch.squeeze(pred3, 1)

            return disp_finetune
            # return [disp_finetune], [pred3]


def PCWNet_G(d=192):
    return PCWNet(d, use_concat_volume=False)


def PCWNet_GC(d=192):
    return PCWNet(d, use_concat_volume=True)
