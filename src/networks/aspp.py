# modified from https://github.com/kazuto1011/deeplab-pytorch/tree/master/libs/models

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.InstanceNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


# class DeepLabV3(nn.Sequential):
#     """
#     DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
#     """

#     def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
#         super(DeepLabV3, self).__init__()

#         # Stride and dilation
#         if output_stride == 8:
#             s = [1, 2, 1, 1]
#             d = [1, 1, 2, 4]
#         elif output_stride == 16:
#             s = [1, 2, 2, 1]
#             d = [1, 1, 1, 2]

#         ch = [64 * 2 ** p for p in range(6)]
#         self.add_module("layer1", _Stem(ch[0]))
#         self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
#         self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
#         self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
#         self.add_module(
#             "layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
#         )
#         self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
#         concat_ch = 256 * (len(atrous_rates) + 2)
#         self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
#         self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))