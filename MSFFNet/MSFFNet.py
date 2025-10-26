import torch
import torch.nn as nn
import torch.nn.functional as F
from LFMB.lfmb import *
from GCMB.WMSA import *
from GCMB.FEFM import *


class Fusion_model(nn.Module):
    def __init__(self, band=4, num_class=6):
        super(Fusion_model, self).__init__()

        self.lfmb = Local_Feature_Modeling_Branch(
            in_channels=1,
            ori_data_channel=band,
        )

        self.sae = Self_Attention_Encoder(
            # in_chans=band,
            in_chans=1,
            embed_dim=64,
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
            frozen_stages=2,
            out_indices=(0, 1, 2, 3)
        )

        self.FEFM = FEFM(
            in_channels_list=[64, 128, 256, 512],
            cam_mid_channels=64,
            cam_reduction_ratio=1,
            frh_out_channels=64
        )

        self.channel_conv = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1)
        self.channel_conv_fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)

        self.advavg = nn.AdaptiveAvgPool2d(1)

        self.seg_head = nn.Conv2d(32, num_class, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]

        x_3d = self.lfmb(x)
        x_trans_list = self.sae(x[:, -1:, :, :])
        x_trans = x_trans_list[-1]

        x_trans = F.interpolate(self.channel_conv(x_trans), size=(h, w), mode='bilinear', align_corners=False)

        x_trans += F.interpolate \
            (self.channel_conv_fusion(self.FEFM([x_trans_list[i] for i in range(len(x_trans_list))])),
                                 size=(h, w),
                                 mode='bilinear',
                                 align_corners=False)

        out = torch.cat([x_3d, x_trans], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        out = self.seg_head(out1 + out2 + x_3d)

        return out
