import torch.nn as nn


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.downsample_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.downsample_layer(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, mode='Conv'):
        super(DownSampling, self).__init__()
        self.mode = mode

        self.downsample_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        )

        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.channels_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=in_channels)

    def forward(self, x):
        if self.mode == 'Conv':
            return self.downsample_layer(x)

        elif self.mode == 'Maxpool':
            return self.channels_conv(self.maxpool3d(x))