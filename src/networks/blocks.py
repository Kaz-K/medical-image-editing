import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_output_act=True):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.double_conv = DoubleConv(in_channels, out_channels, use_output_act=use_output_act)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_output_act=True):
        super(DoubleConv, self).__init__()

        if use_output_act:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )

    def forward(self, x):
        return self.double_conv(x)


class StyledDenorm(nn.Module):
    # original: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py

    def __init__(self,
                 in_channels,
                 style_channels,
                 ) -> None:
        super(StyledDenorm, self).__init__()

        self.param_free_norm = nn.BatchNorm2d(in_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            conv3x3(style_channels, in_channels),
            nn.ReLU(inplace=True),
        )

        self.mlp_gamma = conv3x3(in_channels, in_channels)
        self.mlp_beta = conv3x3(in_channels, in_channels)

    def forward(self, x, style):
        normalized = self.param_free_norm(x)

        actv = self.mlp_shared(style)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class StyledResUpBlock(nn.Module):
    def __init__(self, in_channels, style_channels, out_channels,
                 use_output_act=True,
                 use_pixel_shuffle=False,
                 ):
        super(StyledResUpBlock, self).__init__()

        if use_pixel_shuffle:
            self.up_sample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = StyledDenorm(out_channels, style_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = StyledDenorm(out_channels, style_channels)
        self.act2 = nn.ReLU(inplace=True) if use_output_act else lambda x: x

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        s = self.conv(x)

        x = self.conv1(x)
        x = self.norm1(x, skip_input)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, skip_input)
        x = self.act2(x)

        return s + x
