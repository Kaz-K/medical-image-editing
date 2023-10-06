import torch
import random
import numpy as np
import kornia as K
import torch.nn as nn

from kornia.augmentation import RandomAffine
from kornia.augmentation import RandomHorizontalFlip


class ExpandChannelDim(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image):
        return image.expand(image.size(0), 3, image.size(2), image.size(3))


class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        image = torch.from_numpy(image).float()

        sample.update({
            'image': image,
        })

        return sample


class SqueezeAxis(object):

    def __call__(self, sample):
        image = sample['image']

        if image.ndim == 4:
            assert image.size(0) == 1
            image = image.squeeze(0)

        sample.update({
            'image': image,
        })

        return sample

class NormalizeIntensity(object):

    def __init__(self, vmin=0, vmax=255):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, sample):
        image = sample['image']

        image = torch.clamp(image, min=self.vmin, max=self.vmax)
        image -= self.vmin
        image /= (self.vmax - self.vmin)
        image *= 2.0
        image -= 1.0

        sample.update({
            'image': image,
        })

        return sample


class BaseTransform(object):

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = sample['image']

        needs_squeeze = False
        if image.ndim == 3:
            needs_squeeze = True

        if needs_squeeze:
            image = image.unsqueeze(0)

        params = self.transform.forward_parameters(image.shape)
        image = self.transform(image, params)

        if needs_squeeze:
            image = image.squeeze(0)

        sample.update({
            'image': image,
        })

        return sample


class RandomAffineTransform(BaseTransform):

    def __init__(self,
                 p,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 resample='BILINEAR',
                 ):
        super().__init__(
            transform=RandomAffine(
                p=p,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                resample=resample,
            )
        )


class RandomHorizontalFlipTransform(BaseTransform):

    def __init__(self, p):
        super().__init__(
            transform=RandomHorizontalFlip(p=p),
        )
