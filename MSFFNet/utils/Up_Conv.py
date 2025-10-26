import torch.nn as nn


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, mode='Conv'):
        super(UpSampling, self).__init__()
        self.mode = mode

        self.upsample_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )

        self.channels_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=out_channels)

    def forward(self, x):
        if self.mode == 'Conv':
            return self.upsample_layer(x)

        elif self.mode == 'interpolate':
            x = nn.functional.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            return self.channels_conv(x)