import torch
import torch.nn as nn

from .vq import VQ
from .blocks import ResBlock
from .blocks import UpBlock
from .blocks import DoubleConv
from .initialize import init_weights


class VQVNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 64,
                 filters: list = [64, 128, 256, 512, 1024],
                 dict_size: int = 8,
                 knn_backend: str = 'torch',
                 ):
        super().__init__()

        self.down_conv1_1 = ResBlock(in_channels, filters[0])
        self.down_conv1_2 = ResBlock(filters[0], filters[1])
        self.down_conv1_3 = ResBlock(filters[1], filters[2])
        self.down_conv1_4 = ResBlock(filters[2], filters[3])

        self.double_conv1 = DoubleConv(filters[3], filters[4])

        self.up_conv1_4 = UpBlock(filters[3] + filters[4], filters[3])
        self.up_conv1_3 = UpBlock(filters[2] + filters[3], filters[2])
        self.up_conv1_2 = UpBlock(filters[1] + filters[2], filters[1])
        self.up_conv1_1 = UpBlock(filters[1] + filters[0], filters[0])

        self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        self.vq = VQ(emb_dim=out_channels,
                     dict_size=dict_size,
                     momentum=0.99,
                     eps=1e-5,
                     knn_backend=knn_backend)

        init_weights(self, 'kaiming')

    @property
    def name(self):
        return 'VQVNet'

    def forward(self, x):
        x, skip1_1_out = self.down_conv1_1(x)
        x, skip1_2_out = self.down_conv1_2(x)
        x, skip1_3_out = self.down_conv1_3(x)
        x, skip1_4_out = self.down_conv1_4(x)
        x = self.double_conv1(x)
        x = self.up_conv1_4(x, skip1_4_out)
        x = self.up_conv1_3(x, skip1_3_out)
        x = self.up_conv1_2(x, skip1_2_out)
        x = self.up_conv1_1(x, skip1_1_out)

        x = self.conv_last(x)

        x, commit_loss, ids = self.vq(x)
        ids = torch.transpose(ids, 1, 2)

        return {
            'embed': x,
            'commit_loss': commit_loss,
            'ids': ids,
        }
