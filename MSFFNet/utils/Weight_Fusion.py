import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Weights_Fusion_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, use_conv=True, eps=1e-8):
        super(Weights_Fusion_Block, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=in_channels)
            self.norm = nn.InstanceNorm3d(out_channels)
            self.act = nn.LeakyReLU()

    def forward(self, x, en_x):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        out = fuse_weights[0] * en_x + fuse_weights[1] * x
        if self.use_conv:
            out = self.act(self.norm(self.conv(out)))
        return out


class Feature_Fusion(nn.Module):
    def __init__(self, out_c, n_class):
        super(Feature_Fusion, self).__init__()
        self.transposeconv1 = nn.ConvTranspose3d(out_c, n_class, 3, 1, 1)
        self.pwconv1 = nn.Conv3d(n_class, 4 * n_class, kernel_size=1, groups=n_class)
        self.dwconv = nn.Conv3d(4 * n_class, 4 * n_class, kernel_size=7, padding=3, groups=4 * n_class)
        self.norm = nn.BatchNorm3d(4 * n_class)
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Conv3d(4 * n_class, n_class, kernel_size=1, groups=n_class)

    def forward(self, x):
        x_left = self.transposeconv1(x)
        x_right = self.pwconv1(x_left)
        x_right = self.dwconv(x_right)
        x_right = self.norm(x_right)
        x_right = self.act(x_right)
        x_right = self.pwconv2(x_right)
        x = x_left + x_right
        return x