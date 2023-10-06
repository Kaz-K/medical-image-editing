# modified the original source codes (https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py)
import functools
import torch
import torch.nn as nn

from .actnorm import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 n_filters=64,
                 n_layers=3,
                 # use_actnorm=False,
                 normalization='batchnorm'):
        """Construct a PatchGAN discriminator
        Parameters:
            in_channels (int) -- the number of channels in input images
            n_filters (int)   -- the number of filters in the last conv layer
            n_layers (int)    -- the number of conv layers in the discriminator
            norm_layer        -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        assert normalization in {'instancenorm', 'batchnorm', 'actnorm'}

        if normalization == 'instancenorm':
            norm_layer = nn.InstanceNorm2d

        elif normalization == 'batchnorm':
            norm_layer = nn.BatchNorm2d

        elif normalization == 'actnorm':
            norm_layer = ActNorm

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d  # use_bias = True when nn.InstanceNorm2d is used

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, n_filters, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(n_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(n_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(n_filters * nf_mult, out_channels, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        weights_init(self)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
