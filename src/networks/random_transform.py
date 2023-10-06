import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as K

from dataio import ExpandChannelDim


class RandomTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.expand_channel_dim = ExpandChannelDim()

        self.geometrics = []
        self.photometrics = []

        for module in config.modules:
            if module == 'RandomHorizontalFlip':
                self.geometrics.append(
                    K.augmentation.RandomHorizontalFlip(
                        p=config.RandomHorizontalFlip.p,
                        return_transform=True,
                    )
                )
            elif module == 'RandomAffine':
                self.geometrics.append(
                    K.augmentation.RandomAffine(
                        degrees=config.RandomAffine.degrees,
                        translate=config.RandomAffine.translate,
                        shear=config.RandomAffine.shear,
                        p=config.RandomAffine.p,
                        return_transform=True,
                    )
                )
            elif module == 'ColorJitter':
                self.photometrics.append(
                    K.augmentation.ColorJitter(
                        brightness=config.ColorJitter.brightness,
                        contrast=config.ColorJitter.contrast,
                        saturation=config.ColorJitter.saturation,
                        hue=config.ColorJitter.hue,
                        p=config.ColorJitter.p,
                        return_transform=False,
                    )
                )
            elif module == 'RandomGaussianBlur':
                self.photometrics.append(
                    K.augmentation.RandomGaussianBlur(
                        (config.RandomGaussianBlur.kernel, config.RandomGaussianBlur.kernel),
                        (config.RandomGaussianBlur.sigma, config.RandomGaussianBlur.sigma),
                        p=config.RandomGaussianBlur.p,
                        return_transform=False,
                    )
                )
            elif module == 'RandomPosterize':
                self.photometrics.append(
                    K.augmentation.RandomPosterize(
                        bits=config.RandomPosterize.bits,
                        p=config.RandomPosterize.p,
                        return_transform=False,
                    )
                )
            elif module == 'RandomGaussianNoise':
                self.photometrics.append(
                    K.augmentation.RandomGaussianNoise(
                        std=config.RandomGaussianNoise.std,
                        p=config.RandomGaussianNoise.p,
                        return_transform=False,
                    )
                )

        self.rgb_to_grayscale = K.color.RgbToGrayscale()

    def forward(self, x):
        x = self.expand_channel_dim(x)

        self._transforms = []
        for module in self.geometrics:
            x, transform = module(x)
            self._transforms.append(transform)

        clear_x = x.detach().clone()
        for module in self.photometrics:
            x = module(x)

        x = self.rgb_to_grayscale(x)
        clear_x = self.rgb_to_grayscale(clear_x)

        return x, clear_x

    def forward_transform(self, x):
        x = x.unsqueeze(1)
        dsize = (x.shape[2], x.shape[3])

        for transform in self._transforms:
            x = K.geometry.transform.warp_perspective(x, transform, dsize=dsize, mode='nearest')

        x = x.squeeze(1)
        return x

    def reverse_transform(self, x):
        x = x.unsqueeze(1).float()
        dsize = (x.shape[2], x.shape[3])

        for transform in reversed(self._transforms):
            inv_transform = torch.inverse(transform)
            x = K.geometry.transform.warp_perspective(x, inv_transform, dsize=dsize, mode='nearest')

        x = x.squeeze(1)
        return x
