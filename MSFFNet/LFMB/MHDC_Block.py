import torch
import torch.nn as nn
from timm.models.layers import DropPath


class MHDC_module_S(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_list=[1, 2, 3], drop_path=0.2):
        super(MHDC_module_S, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inter_channels = in_channels // 4
        self.out_inter_channels = out_channels // 4

        self.scale1 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=1,
                                        bias=False
                                              ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
                                    )
        self.scale2 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[0] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[0],
                                        bias=False
                                              ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
                                    )
        self.scale3 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[1] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[1],
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )
        self.scale4 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[2] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[2],
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )

        self.scale_process = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels * 3,
                                        self.out_inter_channels * 3,
                                        kernel_size=3,
                                        padding=1,
                                        groups=3,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels * 3),
                                    nn.LeakyReLU()
        )

        self.compression = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels * 4,
                                        self.out_inter_channels * 4,
                                        kernel_size=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels * 4),
                                    nn.LeakyReLU()
        )

        self.shortcut = nn.Sequential(
                                    nn.Conv3d(
                                        self.in_channels,
                                        self.out_channels,
                                        kernel_size=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels * 4),
                                    nn.LeakyReLU()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        scale_list = []
        x1 = x[:, 0:self.inter_channels, ...]
        x2 = x[:, self.inter_channels:self.inter_channels * 2, ...]
        x3 = x[:, self.inter_channels * 2:self.inter_channels * 3, ...]
        x4 = x[:, self.inter_channels * 3:self.inter_channels * 4, ...]

        x_1 = self.scale1(x1)
        scale_list.append(self.scale2(x2) + x_1)
        scale_list.append(self.scale3(x3) + x_1)
        scale_list.append(self.scale4(x4) + x_1)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_1, scale_out], 1)) + self.shortcut(self.drop_path(x))

        return out



class MHDC_module_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_list=[1, 2, 3], drop_path=0.2):
        super(MHDC_module_D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inter_channels = in_channels // 4
        self.out_inter_channels = out_channels // 4

        self.scale1 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=1,
                                        bias=False
                                              ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
                                    )
        self.scale2 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[0] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[0],
                                        bias=False
                                              ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
                                    )
        self.scale3 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[1] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[1],
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )
        self.scale4 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=dilation_list[2] * (kernel_size - 1) // 2,
                                        dilation=dilation_list[2],
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )

        self.process2 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        padding=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )
        self.process3 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        padding=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )
        self.process4 = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels,
                                        self.out_inter_channels,
                                        kernel_size=kernel_size,
                                        padding=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels),
                                    nn.LeakyReLU()
        )

        self.compression = nn.Sequential(
                                    nn.Conv3d(
                                        self.inter_channels * 4,
                                        self.out_inter_channels * 4,
                                        kernel_size=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels * 4),
                                    nn.LeakyReLU()
        )

        self.shortcut = nn.Sequential(
                                    nn.Conv3d(
                                        self.in_channels,
                                        self.out_channels,
                                        kernel_size=1,
                                        bias=False
                                    ),
                                    nn.InstanceNorm3d(self.out_inter_channels * 4),
                                    nn.LeakyReLU()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        scale_list = []
        x1 = x[:, 0:self.inter_channels, ...]
        x2 = x[:, self.inter_channels:self.inter_channels * 2, ...]
        x3 = x[:, self.inter_channels * 2:self.inter_channels * 3, ...]
        x4 = x[:, self.inter_channels * 3:self.inter_channels * 4, ...]

        x_1 = self.scale1(x1)
        scale_list.append(x_1)
        scale_list.append(self.process2(self.scale2(x2) + scale_list[0]))
        scale_list.append(self.process3(self.scale3(x3) + scale_list[1]))
        scale_list.append(self.process4(self.scale4(x4) + scale_list[2]))

        out = self.compression(torch.cat(scale_list, 1)) + self.shortcut(self.drop_path(x))

        return out