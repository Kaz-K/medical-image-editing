import torch
import torch.nn as nn

from kmeans_pytorch import kmeans

from .vq import VQ
from .blocks import ResBlock
from .blocks import UpBlock
from .blocks import DoubleConv
from .blocks import StyledResUpBlock
from .dropblock import LinearScheduler
from .dropblock import DropBlock2D
from .initialize import init_weights


class UNetEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 filters: list = [64, 128, 256, 512, 1024],
                 dict_size: int = 512,
                 momentum: float = 0.99,
                 knn_backend: str = 'torch',
                 use_styled_up_block: bool = False,
                 num_gpus: int = 4,
                 init_embed: bool = False,
                 ):
        super().__init__()

        self.dict_size = dict_size
        self.init_embed = init_embed
        self.dims = filters[0]
        self.num_gpus = num_gpus

        self.down_conv1_1 = ResBlock(in_channels, filters[0])
        self.down_conv1_2 = ResBlock(filters[0], filters[1])
        self.down_conv1_3 = ResBlock(filters[1], filters[2])
        self.down_conv1_4 = ResBlock(filters[2], filters[3])

        self.double_conv1 = DoubleConv(filters[3], filters[4])

        if use_styled_up_block:
            self.up_conv1_4 = StyledResUpBlock(filters[4], filters[3], filters[3])
            self.up_conv1_3 = StyledResUpBlock(filters[3], filters[2], filters[2])
            self.up_conv1_2 = StyledResUpBlock(filters[2], filters[1], filters[1])
            self.up_conv1_1 = StyledResUpBlock(filters[0], filters[0], filters[0])

        else:
            self.up_conv1_4 = UpBlock(filters[3] + filters[4], filters[3])
            self.up_conv1_3 = UpBlock(filters[2] + filters[3], filters[2])
            self.up_conv1_2 = UpBlock(filters[1] + filters[2], filters[1])
            self.up_conv1_1 = UpBlock(filters[1] + filters[0], filters[0])

        self.vq = VQ(emb_dim=filters[0],
                     dict_size=self.dict_size,
                     momentum=momentum,
                     eps=1e-5,
                     knn_backend=knn_backend)

        init_weights(self, 'kaiming')

    @property
    def name(self):
        return 'UNetEncoder'

    def initialize_embed(self, embed, rank):
        tensor_list = [torch.zeros_like(embed) for _ in range(self.num_gpus)]
        torch.distributed.all_gather(tensor_list, embed)

        if rank == 0:
            with torch.no_grad():
                embed = torch.cat(tensor_list, dim=0)
                embed = embed.detach().clone()
                embed = embed.permute(1, 0, 2, 3).contiguous().view(self.dims, -1)
                embed = torch.t(embed)

                cluster_ids_x, cluster_centers = kmeans(
                    X=embed,
                    num_clusters=self.dict_size,
                    distance='euclidean',
                    device=embed.get_device()
                )

                cluster_centers = cluster_centers.type_as(embed)
                self.vq.embed = cluster_centers.detach()
                torch.distributed.broadcast(self.vq.embed, 0)

        else:
            torch.distributed.broadcast(self.vq.embed, 0)

        self.init_embed = True

    def feature_extraction(self, x):
        x, skip1_1_out = self.down_conv1_1(x)
        x, skip1_2_out = self.down_conv1_2(x)
        x, skip1_3_out = self.down_conv1_3(x)
        x, skip1_4_out = self.down_conv1_4(x)
        x = self.double_conv1(x)
        x = self.up_conv1_4(x, skip1_4_out)
        x = self.up_conv1_3(x, skip1_3_out)
        x = self.up_conv1_2(x, skip1_2_out)
        x = self.up_conv1_1(x, skip1_1_out)
        return x

    def forward(self, x, skip_vq=False, rank=False):
        x = self.feature_extraction(x)
        if skip_vq:
            return x

        else:
            if not self.init_embed:
                self.initialize_embed(x, rank)

            x, commit_loss, ids = self.vq(x)
            ids = torch.transpose(ids, 1, 2)
            ids += 1

            return x, commit_loss, ids

    def get_embed_from_ids(self, ids):
        ids = torch.transpose(ids, 1, 2)
        x = self.vq.lookup(ids).transpose(1, -1)
        return x
