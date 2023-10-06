from collections import OrderedDict

import torch
import torch.nn as nn

from .vq import VQ
from .blocks import ResBlock
from .blocks import UpBlock
from .blocks import DoubleConv
from .blocks import StyledResUpBlock
from .dropblock import LinearScheduler
from .dropblock import DropBlock2D
from .initialize import init_weights
from .aspp import ASPP


class UNetDecoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filters: list = [64, 128, 256, 512, 1024],
                 use_dropblock: bool = False,
                 block_size: int = 30,
                 start_value: float = 0.3,
                 stop_value: float = 0.9,
                 nr_steps: int = 100,
                 dropped_skip_layers: list = [5, 6],
                 use_styled_up_block: bool = True,
                 use_pixel_shuffle: bool = True,
                 use_last_pixel_shuffle: bool = False,
                 ):
        super().__init__()

        assert use_styled_up_block
        self.use_last_pixel_shuffle = use_last_pixel_shuffle
        self.dropped_skip_layers = dropped_skip_layers

        if use_dropblock:
            self.dropblock = LinearScheduler(
                DropBlock2D(block_size=block_size, drop_prob=start_value),
                start_value=start_value,
                stop_value=stop_value,
                nr_steps=nr_steps,
            )
        else:
            self.dropblock = lambda x: x

        self.down_convs = []
        for i in range(len(filters) - 1):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters[i - 1]

            out_ch = filters[i]

            self.add_module('down_conv2_{}'.format(str(i + 1)),
                            ResBlock(in_ch, out_ch))

            self.down_convs.append(getattr(self, 'down_conv2_{}'.format(str(i + 1))))

        self.double_conv2 = DoubleConv(filters[i], filters[i + 1])

        self.up_convs = []
        if use_last_pixel_shuffle:
            self.pixel_shuffles = []

        for i in reversed(range(len(filters) - 1)):
            in_ch = filters[i + 1]
            out_ch = filters[i]

            self.add_module('up_conv2_{}'.format(i + 1),
                            StyledResUpBlock(in_ch, out_ch, out_ch, use_pixel_shuffle=use_pixel_shuffle))

            self.up_convs.append(getattr(self, 'up_conv2_{}'.format(i + 1)))

            if use_last_pixel_shuffle:
                if i > 0:
                    self.add_module('pixel_shuffle2_{}'.format(i + 1),
                                    nn.Sequential(
                                        nn.Conv2d(out_ch, (4 ** i) * filters[0], 3, 1, 1),
                                        nn.PixelShuffle(2 ** i),
                                    ))

                    self.pixel_shuffles.append(getattr(self, 'pixel_shuffle2_{}'.format(i + 1)))

        init_weights(self, 'kaiming')

        if use_last_pixel_shuffle:
            self.conv_last = nn.Conv2d((len(filters) - 1) * filters[0], out_channels, kernel_size=1)
        else:
            # self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            
            # self.conv_last = nn.Sequential(
            #     DoubleConv(filters[0], 4 * filters[0]),
            #     DoubleConv(4 * filters[0], 4 * filters[0]),
            #     nn.Conv2d(4 * filters[0], out_channels, kernel_size=1),
            # )

            self.conv_last = nn.Sequential(
                ASPP(filters[0], filters[0], [2, 6, 12, 18]),
                DoubleConv(5 * filters[0], filters[0]),
            )
            self.conv1x1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        self.final_act = nn.Tanh()

        # init_weights(self, 'kaiming')

    @property
    def name(self):
        return 'UNetDecoder'

    def forward(self, x):
        d_skips = []

        for d in self.down_convs:
            x, d_skip = d(x)
            d_skips.append(d_skip)

        x = self.double_conv2(x)

        d_skips.reverse()

        if self.use_last_pixel_shuffle:
            xs = []
            for i, (u, d_skip) in enumerate(zip(self.up_convs, d_skips)):

                if i in self.dropped_skip_layers:
                    d_skip = torch.zeros_like(d_skip)
                else:
                    d_skip = self.dropblock(d_skip)

                x = u(x, d_skip)
                xs.append(x)

            self.pixel_shuffles.append(lambda x: x)

            out = []
            for x, ps in zip(xs, self.pixel_shuffles):
                o = ps(x)
                out.append(o)

            out.reverse()
            out = torch.cat(out, dim=1)
            out = self.conv_last(out)

        else:
            for i, (u, d_skip) in enumerate(zip(self.up_convs, d_skips)):

                if i in self.dropped_skip_layers:
                    d_skip = torch.zeros_like(d_skip)
                else:
                    d_skip = self.dropblock(d_skip)

                x = u(x, d_skip)

            out = x + self.conv_last(x)
            out = self.conv1x1(out)
            # out = self.conv_last(x)

        recon = self.final_act(out)
        return recon
