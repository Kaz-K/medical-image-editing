import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vq import VQ


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32,
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 use_conv_shortcut=False,
                 p_dropout=0.0,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(p_dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)

            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.norm = Normalize(in_channels)

        self.q = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.k = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.v = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.proj_out = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)      # (b, hw, c)
        k = k.reshape(b, c, h * w)  # (b, c, hw)
        w_ = torch.bmm(q, k)        # (b, hw, hw)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)  # (b, c, hw)
        w_ = w_.permute(0, 2, 1)    # (b, hw, hw)
        h_ = torch.bmm(v, w_)       # (b, c, hw)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 ch_multiplier,
                 num_res_blocks,
                 attn_resolutions,
                 resolution,
                 p_dropout,
                 resamp_with_conv,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.num_resolutions = len(ch_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        curr_res = resolution
        in_ch_multiplier = (1,) + tuple(ch_multiplier)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = mid_channels * in_ch_multiplier[i_level]
            block_out = mid_channels * ch_multiplier[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         p_dropout=p_dropout))

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(in_channels=block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_channels=block_in,
                                             with_conv=resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       p_dropout=p_dropout)
        self.mid.attn_1 = AttnBlock(in_channels=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       p_dropout=p_dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 ch_multiplier,
                 num_res_blocks,
                 attn_resolutions,
                 resolution,
                 p_dropout,
                 resamp_with_conv,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.num_resolutions = len(ch_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        curr_res = resolution // 2**(self.num_resolutions - 1)

        in_ch_multiplier = (1,) + tuple(ch_multiplier)
        block_in = mid_channels * ch_multiplier[-1]

        self.conv_in = nn.Conv2d(in_channels,
                                 block_in,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       p_dropout=p_dropout)
        self.mid.attn_1 = AttnBlock(in_channels=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       p_dropout=p_dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = mid_channels * ch_multiplier[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         p_dropout=p_dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(in_channels=block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(in_channels=block_in,
                                       with_conv=resamp_with_conv)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        recon = self.conv_out(h)

        return recon


class VQGAN(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 mid_channels: int = 32,
                 out_channels: int = 9,
                 emb_dim: int = 512,
                 dict_size: int = 64,
                 enc_ch_multiplier: tuple = (1, 2, 4, 8, 16, 32),
                 dec_ch_multiplier: tuple = (1, 1, 2, 4, 8, 16),
                 num_res_blocks: int = 2,
                 enc_attn_resolutions: list = [],
                 dec_attn_resolutions: list = [16],
                 resolution: int = 512,
                 p_dropout: float = 0.0,
                 resamp_with_conv: bool = True,
                 knn_backend: str = 'torch',
                 ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=emb_dim,
            ch_multiplier=enc_ch_multiplier,
            num_res_blocks=num_res_blocks,
            attn_resolutions=enc_attn_resolutions,
            resolution=resolution,
            p_dropout=p_dropout,
            resamp_with_conv=resamp_with_conv,
        )

        self.decoder = Decoder(
            in_channels=emb_dim,
            mid_channels=mid_channels,
            out_channels=out_channels,
            ch_multiplier=dec_ch_multiplier,
            num_res_blocks=num_res_blocks,
            attn_resolutions=dec_attn_resolutions,
            resolution=resolution,
            p_dropout=p_dropout,
            resamp_with_conv=resamp_with_conv,
        )

        self.vq = VQ(
            emb_dim=emb_dim,
            dict_size=dict_size,
            momentum=0.99,
            eps=1e-5,
            knn_backend=knn_backend,
        )

    def forward(self, x):
        x = self.encoder(x)
        emb, commit_loss, ids = self.vq(x)
        recon = self.decoder(emb)
        return recon, commit_loss, ids, emb

    def generate_image_from_ids(self, ids):
        x = self.vq.lookup(ids)
        x = x.transpose(3, 1)
        recon = self.decoder(x)
        # x = torch.argmax(x, dim=1)
        return recon
