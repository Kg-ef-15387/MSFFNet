import torch
import torch.nn as nn
from LFMB.MHDC_Block import *
from LFMB.Partial_Conv import *
from utils.Down_Conv import *
from utils.Up_Conv import *
from utils.Weight_Fusion import *


class Local_Feature_Modeling_Branch(nn.Module):
    def __init__(self, in_channels, ori_data_channel=4, n_classes=6, dims=None):
        super(Local_Feature_Modeling_Branch, self).__init__()
        if dims is None:
            dims = [32, 64, 128, 256]

        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=1)

        self.encoder1 = nn.Sequential(
            MHDC_module_S(dims[0], dims[0]),
            MHDC_module_S(dims[0], dims[0], dilation_list=[1, 5, 9])
        )

        self.down1 = DownSampling(dims[0], dims[1], mode='Conv')

        self.encoder2 = nn.Sequential(
            MHDC_module_D(dims[1], dims[1], dilation_list=[1, 2, 3]),
            MHDC_module_D(dims[1], dims[1], dilation_list=[1, 2, 3])
        )

        self.down2 = DownSampling(dims[1], dims[2], mode='Conv')

        self.encoder3 = nn.Sequential(DoubleConv(dims[2]), DoubleConv(dims[2]))
        self.down3 = DownSampling(dims[2], dims[3], mode='Conv')

        self.bottleneck = nn.Sequential(DoubleConv(dims[3]),
                                        DoubleConv(dims[3]),
                                        DoubleConv(dims[3]),
                                        DoubleConv(dims[3]))

        self.up1 = UpSampling(dims[3], dims[2], mode='Conv')
        self.decoder1 = DoubleConv(dims[2])
        self.weight1 = Weights_Fusion_Block(in_channels=dims[2], out_channels=dims[2], use_conv=True)

        self.up2 = UpSampling(dims[2], dims[1], mode='Conv')
        self.decoder2 = nn.Sequential(
            MHDC_module_D(dims[1], dims[1], dilation_list=[1, 2, 3]),
            MHDC_module_D(dims[1], dims[1], dilation_list=[1, 2, 3])
        )
        self.weight2 = Weights_Fusion_Block(in_channels=dims[1], out_channels=dims[1], use_conv=True)

        self.up3 = UpSampling(dims[1], dims[0], mode='Conv')
        self.decoder3 = nn.Sequential(
            MHDC_module_S(dims[0], dims[0]),
            MHDC_module_S(dims[0], dims[0], dilation_list=[1, 5, 9])
        )
        self.weight3 = Weights_Fusion_Block(in_channels=dims[0], out_channels=dims[0], use_conv=True)

        self.ff = Feature_Fusion(dims[0], dims[0])

        self.compress_depth = nn.Sequential(
            nn.Conv3d(dims[0], dims[0], kernel_size=(ori_data_channel, 1, 1), stride=1),
            nn.BatchNorm3d(dims[0]),
            nn.LeakyReLU()
        )
        # self.seg_head = nn.Conv2d(dims[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.encoder1(x)
        out1 = x
        x = self.down1(x)
        x = self.encoder2(x)
        out2 = x
        x = self.down2(x)
        x = self.encoder3(x)
        out3 = x
        x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x_dec = self.weight1(x, out3)
        x = self.decoder1(x_dec)
        x = self.up2(x)
        x_dec = self.weight2(x, out2)
        x = self.decoder2(x_dec)
        x = self.up3(x)
        x_dec = self.weight3(x, out1)
        x = self.decoder3(x_dec)
        x = self.ff(x)
        x = self.compress_depth(x)
        x = x.squeeze(2)

        return x