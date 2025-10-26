import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ASPP import *

class MCAM(nn.Module):
    def __init__(self, in_channels_list, mid_channels, reduction_ratio=1):
        super(MCAM, self).__init__()
        self.in_channels_list = in_channels_list
        self.mid_channels = mid_channels
        self.feature_num = len(in_channels_list)

        self.channels_convs = nn.ModuleList()
        for i in range(self.feature_num):
            channels_conv = nn.Sequential(
                nn.Conv2d(self.in_channels_list[i], self.mid_channels, kernel_size=1),
                nn.BatchNorm2d(self.mid_channels),
                nn.ReLU()
            )

            self.channels_convs.append(channels_conv)

        self.dila_conv = nn.Sequential(
            nn.Conv2d(
                self.mid_channels * self.feature_num // reduction_ratio,
                self.mid_channels // reduction_ratio,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ASPP(
                in_channels=self.mid_channels // reduction_ratio,
                atrous_rates=[1, 2, 5],
                # out_channels=self.mid_channels // (4 * reduction_ratio)
                out_channels=self.mid_channels // reduction_ratio
            ),
            nn.Conv2d(
                self.mid_channels // reduction_ratio,
                self.mid_channels // reduction_ratio,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.mid_channels // reduction_ratio),
            nn.ReLU()
            )

        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        for i in range(self.feature_num):
            self.att_conv.append(
                nn.Conv2d(
                    self.mid_channels // reduction_ratio,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                nn.Conv2d(
                    self.mid_channels // reduction_ratio,
                    self.mid_channels // reduction_ratio,
                    kernel_size=3,
                    stride=down_stride,
                    padding=1))

    def forward(self, feature_list):
        assert len(feature_list) == self.feature_num

        prev_shape = feature_list[0].shape[2:]
        multi_feats = [self.channels_convs[0](feature_list[0])]
        for i in range(1, self.feature_num):
            x = self.channels_convs[i](feature_list[i])
            x = F.interpolate(x, size=prev_shape, mode='nearest')
            multi_feats.append(x)

        multi_feats = torch.cat(multi_feats, 1)
        fusion_fea = self.dila_conv(multi_feats)

        multi_atts = []
        for i in range(self.feature_num):
            fusion_fea = self.down_conv[i](fusion_fea)
            lvl_att = self.att_conv[i](fusion_fea)
            multi_atts.append(self.sigmoid(lvl_att))

        return multi_atts


class Feature_Refinement_Head(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(Feature_Refinement_Head, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.feature_num = len(in_channels_list)

        self.pre_convs = nn.ModuleList()
        for i in range(self.feature_num):
            channels_conv = nn.Sequential(
                nn.Conv2d(self.in_channels_list[i], self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()
            )

            self.pre_convs.append(channels_conv)

        self.weights = nn.Parameter(torch.ones(self.feature_num, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

        self.pa = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.out_channels),
            nn.Sigmoid()
        )
        self.adp_avg = nn.AdaptiveAvgPool2d(1)
        self.adp_max = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.ca = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_channels // 16, self.out_channels, kernel_size=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, groups=self.out_channels),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, feature_list):
        assert len(feature_list) == self.feature_num

        prev_shape = feature_list[0].shape[2:]
        multi_feats = [self.pre_convs[0](feature_list[0])]
        for i in range(1, self.feature_num):
            x = self.pre_convs[i](feature_list[i])
            x = F.interpolate(x, size=prev_shape, mode='nearest')
            multi_feats.append(x)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        fused = sum(fuse_weights[i] * multi_feats[i] for i in range(self.feature_num))
        fused = self.post_conv(fused)
        shortcut = self.shortcut(fused)
        pa = self.pa(fused) * fused
        ca = self.sigmoid(
            self.ca(self.adp_avg(fused)) + self.ca(self.adp_max(fused))
        ) * fused
        fused = pa + ca
        fused = self.proj(fused) + shortcut
        fused = self.act(fused)

        return fused


class FEFM(nn.Module):
    def __init__(self,
                 in_channels_list,
                 cam_mid_channels,
                 cam_reduction_ratio,
                 frh_out_channels
                 ):
        super(FEFM, self).__init__()

        self.CAM = MCAM(
            in_channels_list=in_channels_list,
            mid_channels=cam_mid_channels,
            reduction_ratio=cam_reduction_ratio
        )

        self.FRH = Feature_Refinement_Head(
            in_channels_list=in_channels_list,
            out_channels=frh_out_channels
        )

    def forward(self, feature_list):
        att_list = self.CAM(feature_list)
        en_fea = [(1 + att_list[i]) * feature_list[i] for i in range(len(feature_list))]
        fu_fea = self.FRH(en_fea)

        return fu_fea