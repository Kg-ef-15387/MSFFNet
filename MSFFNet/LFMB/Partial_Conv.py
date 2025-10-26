import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.forward = self.forward_split_cat

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class pconv_block(nn.Module):
    def __init__(self, dim, drop_path=0.2):
        super(pconv_block, self).__init__()
        self.partialconv = Partial_conv3(dim)
        self.norm = nn.InstanceNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.partialconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = pconv_block(dim)
        self.conv2 = pconv_block(dim)

    def forward(self, x):
        return self.conv2(self.conv1(x))