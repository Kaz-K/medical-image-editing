import torch
import torch.nn as nn

from .vq import VQ
from .blocks import ResBlock
from .blocks import UpBlock
from .blocks import DoubleConv
from .dropblock import LinearScheduler
from .dropblock import DropBlock2D
from .initialize import init_weights


class VQWNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filters: list = [64, 128, 256, 512, 1024],
                 dict_size: int = 512,
                 knn_backend: str = 'torch',
                 use_dropblock: bool = False,
                 block_size: int = 30,
                 drop_prob: float = 0.3,
                 nr_steps: int = 100,
                 freeze_first_half: bool = False,
                 ):
        super().__init__()

        assert in_channels == out_channels
        self.freeze_first_half = freeze_first_half

        self.down_conv1_1 = ResBlock(in_channels, filters[0])
        self.down_conv1_2 = ResBlock(filters[0], filters[1])
        self.down_conv1_3 = ResBlock(filters[1], filters[2])
        self.down_conv1_4 = ResBlock(filters[2], filters[3])

        self.double_conv1 = DoubleConv(filters[3], filters[4])

        self.up_conv1_4 = UpBlock(filters[3] + filters[4], filters[3])
        self.up_conv1_3 = UpBlock(filters[2] + filters[3], filters[2])
        self.up_conv1_2 = UpBlock(filters[1] + filters[2], filters[1])
        self.up_conv1_1 = UpBlock(filters[1] + filters[0], filters[0])

        self.vq = VQ(emb_dim=filters[0],
                     dict_size=dict_size,
                     momentum=0.99,
                     eps=1e-5,
                     knn_backend=knn_backend)

        if use_dropblock:
            self.dropblock = LinearScheduler(
                DropBlock2D(block_size=block_size, drop_prob=0.),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=nr_steps,
            )
        else:
            self.dropblock = lambda x: x

        self.down_conv2_1 = ResBlock(filters[0], filters[0])
        self.down_conv2_2 = ResBlock(filters[0], filters[1])
        self.down_conv2_3 = ResBlock(filters[1], filters[2])
        self.down_conv2_4 = ResBlock(filters[2], filters[3])

        self.double_conv2 = DoubleConv(filters[3], filters[4])

        self.up_conv2_4 = UpBlock(filters[3] + filters[4], filters[3])
        self.up_conv2_3 = UpBlock(filters[2] + filters[3], filters[2])
        self.up_conv2_2 = UpBlock(filters[1] + filters[2], filters[1])
        self.up_conv2_1 = UpBlock(filters[1] + filters[0], filters[0])

        self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

        init_weights(self, 'kaiming')

        if self.freeze_first_half:
            self._freeze_first_half()

    @property
    def name(self):
        return 'VQWNet'

    def _freeze_first_half(self):
        self.down_conv1_1.requires_grad = False
        self.down_conv1_2.requires_grad = False
        self.down_conv1_3.requires_grad = False
        self.down_conv1_4.requires_grad = False
        self.double_conv1.requires_grad = False
        self.up_conv1_4.requires_grad = False
        self.up_conv1_3.requires_grad = False
        self.up_conv1_2.requires_grad = False
        self.up_conv1_1.requires_grad = False
        self.vq.requires_grad = False

    def forward(self, x):
        if not self.freeze_first_half:
            x, skip1_1_out = self.down_conv1_1(x)
            x, skip1_2_out = self.down_conv1_2(x)
            x, skip1_3_out = self.down_conv1_3(x)
            x, skip1_4_out = self.down_conv1_4(x)
            x = self.double_conv1(x)
            x = self.up_conv1_4(x, skip1_4_out)
            x = self.up_conv1_3(x, skip1_3_out)
            x = self.up_conv1_2(x, skip1_2_out)
            x = self.up_conv1_1(x, skip1_1_out)

            embed = x
            x, commit_loss, ids = self.vq(x)
            ids = torch.transpose(ids, 1, 2)
            ids += 1

        else:
            with torch.no_grad():
                x, skip1_1_out = self.down_conv1_1(x)
                x, skip1_2_out = self.down_conv1_2(x)
                x, skip1_3_out = self.down_conv1_3(x)
                x, skip1_4_out = self.down_conv1_4(x)
                x = self.double_conv1(x)
                x = self.up_conv1_4(x, skip1_4_out)
                x = self.up_conv1_3(x, skip1_3_out)
                x = self.up_conv1_2(x, skip1_2_out)
                x = self.up_conv1_1(x, skip1_1_out)

                x, commit_loss, ids = self.vq(x)
                ids = torch.transpose(ids, 1, 2)
                ids += 1

                x = x.detach()
                embed = x

        x = self.dropblock(x)

        x, skip2_1_out = self.down_conv2_1(x)
        x, skip2_2_out = self.down_conv2_2(x)
        x, skip2_3_out = self.down_conv2_3(x)
        x, skip2_4_out = self.down_conv2_4(x)
        x = self.double_conv2(x)
        x = self.up_conv2_4(x, skip2_4_out)
        x = self.up_conv2_3(x, skip2_3_out)
        x = self.up_conv2_2(x, skip2_2_out)
        x = self.up_conv2_1(x, skip2_1_out)

        x = self.conv_last(x)
        out = self.final_act(x)

        return {
            'recon': out,
            'embed': embed,
            'commit_loss': commit_loss,
            'ids': ids,
        }

    def generate_images_from_ids(self, ids):
        ids = torch.transpose(ids, 1, 2)

        with torch.no_grad():
            x = self.vq.lookup(ids).transpose(1, -1)

            x, skip2_1_out = self.down_conv2_1(x)
            x, skip2_2_out = self.down_conv2_2(x)
            x, skip2_3_out = self.down_conv2_3(x)
            x, skip2_4_out = self.down_conv2_4(x)
            x = self.double_conv2(x)
            x = self.up_conv2_4(x, skip2_4_out)
            x = self.up_conv2_3(x, skip2_3_out)
            x = self.up_conv2_2(x, skip2_2_out)
            x = self.up_conv2_1(x, skip2_1_out)

            x = self.conv_last(x)
            out = self.final_act(x)

        return {
            'recon': out,
            'ids': ids,
        }
