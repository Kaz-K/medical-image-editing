import os
import torch
import time
import datetime
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import threading
from dotenv import load_dotenv

from trainers.base import CMAP
from utils import save_image
from networks import UNetEncoder
from networks import UNetDecoder
from dataio.transforms import ToTensor
from utils import t_normalize as normalize
from utils import denormalize


load_dotenv()
LUNG_CKPT = os.environ.get("LUNG_CKPT")
LUNG_EDITED_FILE = os.environ.get("LUNG_EDITED_FILE")
CRC_CKPT = os.environ.get("CRC_CKPT")
CRC_EDITED_FILE = os.environ.get("CRC_EDITED_FILE")


class LungConfig:
    config_name = 'LungConfig'
    resume_checkpoint = LUNG_CKPT
    in_channels = 1
    enc_filters = [16, 32, 64, 128, 256]
    dec_filters = [32, 64, 128, 256, 512]
    dict_size = 10
    momentum = 0.999
    knn_backend = 'torch'
    edited_file_path = LUNG_EDITED_FILE
    save_dir_path = 'inference'
    window_width = 4096
    window_center = 0.0
    window_scale = 2.0
    use_dropblock = False
    block_size = 30
    start_value = 0.1
    stop_value = 0.5
    nr_steps = 20
    dropped_skip_layers = []
    use_styled_up_block = True
    use_pixel_shuffle = False


class CRCConfig:
    config_name = 'CRCConfig'
    resume_checkpoint = CRC_CKPT
    in_channels = 1
    enc_filters = [16, 32, 64, 128, 256]
    dec_filters = [32, 64, 128, 256, 512]
    dict_size = 10
    momentum = 0.999
    knn_backend = 'torch'
    edited_file_path = CRC_EDITED_FILE
    save_dir_path = 'inference'
    use_dropblock = False
    block_size = 30
    start_value = 0.1
    stop_value = 0.5
    nr_steps = 20
    dropped_skip_layers = []
    use_styled_up_block = True
    use_pixel_shuffle = False


LUNG_WINDOW = {
    'width': 1500,
    'center': -550,
    'scale': 2.0,
}


def transform(sample):
    return ToTensor()(sample)


def save_as_nifti(data, path):
    data = data.float().cpu().numpy()
    data = data.transpose(1, 0)[::-1, ::-1]
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, path)


def load_from_nifti(path):
    data = nib.load(path).get_fdata()
    if data.ndim == 3:
        data = data[:, :, 0]
    data = data.transpose(1, 0)[::-1, ::-1].copy()
    return data


def init_from_ckpt(path, model, key_name, delete_string='model.'):
    state_dict = torch.load(path, map_location=torch.device('cpu'))[
        'state_dict']
    new_state_dict = state_dict.copy()

    for k in state_dict.keys():
        if not k.startswith(key_name):
            del new_state_dict[k]

        if k.startswith(delete_string):
            del new_state_dict[k]
            new_state_dict[k[len(delete_string):]] = state_dict[k]

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Restored from {path}")


def load_model(config):
    encoder = UNetEncoder(
        in_channels=config.in_channels,
        filters=config.enc_filters,
        dict_size=config.dict_size,
        momentum=config.momentum,
        knn_backend=config.knn_backend,
        use_styled_up_block=False,
        num_gpus=4,
        init_embed=False,
    )

    decoder = UNetDecoder(
        in_channels=config.enc_filters[0],
        out_channels=config.in_channels,
        filters=config.dec_filters,
        use_dropblock=config.use_dropblock,
        block_size=config.block_size,
        start_value=config.start_value,
        stop_value=config.stop_value,
        nr_steps=config.nr_steps,
        dropped_skip_layers=config.dropped_skip_layers,
        use_styled_up_block=True,
        use_pixel_shuffle=config.use_pixel_shuffle,
    )

    init_from_ckpt(config.resume_checkpoint, encoder, 'encoder', 'encoder.')
    init_from_ckpt(config.resume_checkpoint, decoder, 'decoder', 'decoder.')

    encoder.eval()
    decoder.eval()

    return encoder, decoder


def denorm_norm(recon, config):
    recon = denormalize(recon,
                        width=config.window_width,
                        center=config.window_center,
                        scale=config.window_scale)

    recon = normalize(recon,
                      width=LUNG_WINDOW['width'],
                      center=LUNG_WINDOW['center'],
                      scale=LUNG_WINDOW['scale'])

    return recon


if __name__ == '__main__':

    config = LungConfig()
    encoder, decoder = load_model(config)

    def inner(prev_map):
        loaded_map = load_from_nifti(config.edited_file_path).astype(np.int32)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        if prev_map is None or not np.array_equal(prev_map, loaded_map):
            print('[{}] Processing...'.format(timestamp))

            if config.config_name == 'CRCConfig':
                loaded_map = np.flipud(loaded_map).copy()

            map = torch.from_numpy(
                loaded_map).long().unsqueeze(0)  # (1, 512, 512)

            mask = torch.zeros_like(map)
            mask[map == 0] = 1
            map[mask == 1] = 1
            mask = 1 - mask
            map -= 1

            with torch.no_grad():
                embed = encoder.get_embed_from_ids(map)  # (1, 16, 512, 512)

                embed = embed * mask[:, None, :, :]
                embed = embed * mask.numel() / mask.sum()

                recon = decoder(embed)

                if config.config_name == 'LungConfig':
                    recon = denorm_norm(recon, config)

                recon = recon[0, 0, ...].detach().cpu().numpy()
                map = map[0, ...].detach().cpu().numpy()
                mask = mask[0, ...].detach().cpu().numpy()

                if config.config_name == 'CRCConfig':
                    recon = np.flipud(recon).copy()
                    map = np.flipud(map).copy()
                    mask = np.flipud(mask).copy()

            plt.imshow(recon, cmap='gray', vmin=-1, vmax=1)
            plt.axis('off')
            plt.show()
            plt.clf()

            map += 1
            map[mask == 0] = 0

            save_file_name = config.edited_file_path.split('.')[
                0] + '_' + timestamp

            save_image(recon, 'gray', -1, 1,
                       'recon_' + save_file_name + '_img.png')
            save_image(map, CMAP, 0, 10,
                       'label_' + save_file_name + '_lbl.png')

            return map.astype(np.int32)

        else:
            print('[{}] Skip...'.format(timestamp))
            return prev_map

    prev_map = None

    while True:
        try:
            prev_map = inner(prev_map)
        except Exception as e:
            print(e.args)

        time.sleep(1)
