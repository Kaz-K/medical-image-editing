import os
import numpy as np
import nibabel as nib
import json
import torch
import collections
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.nn.utils import spectral_norm

from .logger import ModelSaver
from .logger import Logger
from .init_seed import InitSeedAndSaveConfig


def normalize(image, width=1500, center=-550, scale=2.0):
    vmax = center + width // 2
    vmin = center - width // 2

    image = np.clip(image, a_min=vmin, a_max=vmax)
    image -= vmin
    image /= (vmax - vmin)
    image -= 0.5
    image *= scale

    return image


def t_normalize(image, width=1500, center=-550, scale=2.0):
    vmax = center + width // 2
    vmin = center - width // 2

    # image = torch.clamp(image, min=vmin, max=vmax)
    image = image - vmin
    image = image / (vmax - vmin)
    image = image - 0.5
    image = image * scale

    return image


def denormalize(image, width, center, scale):
    vmax = center + width // 2
    vmin = center - width // 2

    image = image / scale
    image = image + 0.5
    image = image * (vmax - vmin)
    image = image + vmin
    return image


def apply_spectral_norm(net):
    def _add_spectral_norm(m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            m = spectral_norm(m)
        elif classname.find('Linear') != -1:
            m = spectral_norm(m)

    print('applying normalization [spectral_norm]')
    net.apply(_add_spectral_norm)


def to_image(tensor, is_ids=False, retain_batch=False):
    if retain_batch:
        if is_ids:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()[:, 0, ...]

    else:
        if is_ids:
            return tensor.detach().cpu().numpy()[0, ...]
        else:
            return tensor.detach().cpu().numpy()[0, 0, ...]


def denorm(array, vmin, vmax):
    array += 1.0
    array /= 2.0
    array *= (vmax - vmin)
    array += vmin
    return array


def norm(array):
    array *= 2.0
    array -= 1.0
    return array


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_json(path):
    def _json_object_hook(d):
        for k, v in d.items():
            d[k] = None if v is False else v
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


def save_images(image, image_1, image_2, recon_1, ids_1, recon_2, ids_2):
    batch_size = image.size(0)

    image = to_cpu(image)[:, 0, ...]
    image_1 = to_cpu(image_1)[:, 0, ...]
    image_2 = to_cpu(image_2)[:, 0, ...]
    recon_1 = to_cpu(recon_1)[:, 0, ...]
    recon_2 = to_cpu(recon_2)[:, 0, ...]
    ids_1 = to_cpu(ids_1)
    ids_2 = to_cpu(ids_2)

    for i in range(batch_size):
        img = image[i, ...]
        img_1 = image_1[i, ...]
        img_2 = image_2[i, ...]
        rec_1 = recon_1[i, ...]
        rec_2 = recon_2[i, ...]
        i_1 = ids_1[i, ...]
        i_2 = ids_2[i, ...]

        plt.subplot(1, 7, 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
        plt.subplot(1, 7, 2)
        plt.axis('off')
        plt.imshow(img_1, cmap='gray', vmin=-1, vmax=1)
        plt.subplot(1, 7, 3)
        plt.axis('off')
        plt.imshow(img_2, cmap='gray', vmin=-1, vmax=1)
        plt.subplot(1, 7, 4)
        plt.axis('off')
        plt.imshow(rec_1, cmap='gray', vmin=-1, vmax=1)
        plt.subplot(1, 7, 5)
        plt.axis('off')
        plt.imshow(rec_2, cmap='gray', vmin=-1, vmax=1)
        plt.subplot(1, 7, 6)
        plt.axis('off')
        plt.imshow(i_1, cmap=CMAP, vmin=0, vmax=7)
        plt.subplot(1, 7, 7)
        plt.axis('off')
        plt.imshow(i_2, cmap=CMAP, vmin=0, vmax=7)
        plt.savefig('temp_{}.png'.format(str(i)), bbox_inches='tight', dpi=300)
        plt.clf()


def save_image(image, cmap, vmin, vmax, path):
    plt.axis('off')
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.clf()


def save_fused_image(image1, cmap1, vmin1, vmax1, image2, cmap2, vmin2, vmax2, alpha, path):
    plt.axis('off')
    plt.imshow(image1, cmap=cmap1, vmin=vmin1, vmax=vmax1)
    plt.imshow(image2, cmap=cmap2, vmin=vmin2, vmax=vmax2, alpha=alpha)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.clf()


def subplot_image(image, title, cmap, vmin, vmax, x, y, z, fontsize=5):
    plt.subplot(x, y, z)
    plt.axis('off')
    plt.gca().title.set_fontsize(fontsize)
    plt.gca().title.set_text(title)

    if vmin is None:
        vmin = image.min()

    if vmax is None:
        vmax = image.max()

    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)


def cutmix_coordinates(height, width, alpha=1.):
    # https://github.com/lucidrains/unet-stylegan2/blob/bae736c55072d89566059ec91d1b160500b86090/unet_stylegan2/unet_stylegan2.py#L198
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam


def cutmix(source, target, coors, alpha=1.):
    # https://github.com/lucidrains/unet-stylegan2/blob/bae736c55072d89566059ec91d1b160500b86090/unet_stylegan2/unet_stylegan2.py#L198
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source


def mask_src_tgt(source, target, mask):
    # https://github.com/lucidrains/unet-stylegan2/blob/bae736c55072d89566059ec91d1b160500b86090/unet_stylegan2/unet_stylegan2.py#L198
    return source * mask + (1 - mask) * target


def to_nifti(array):
    array = np.transpose(array)[::-1, ::-1]
    return nib.Nifti1Image(array, affine=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))
