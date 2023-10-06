import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from focal_frequency_loss import FocalFrequencyLoss as FFL

from torchmetrics import MeanSquaredError as NMSE_metric
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM_metric
from torchmetrics import PeakSignalNoiseRatio as PSNR_metric

from networks import UNetEncoder
from networks import UNetDecoder
from networks import NLayerDiscriminator
from networks import RandomTransform
from networks import UNetDiscriminator
from networks import VQGAN
from functions import EmbeddingLoss
from functions import OneHotEncoder
from functions import VGGLoss
from functions import LPIPSLoss
from utils import apply_spectral_norm
from utils import t_normalize as normalize
from utils import denormalize
from dataio import get_data_loader


CMAP = "Spectral"

SNAPSHOT_INTERVAL = 100

LUNG_WINDOW = {
    'width': 1500,
    'center': -550,
    'scale': 2.0,
}

MEDIASTINAL_WINDOW = {
    'width': 400,
    'center': 20,
    'scale': 2.0,
}


def getattr_else_none(config, attr):
    if hasattr(config, attr):
        return getattr(config, attr)
    else:
        return None


class TrainerBase(pl.LightningModule):

    def __init__(self, config, save_dir_path, monitoring_metrics,
                 uploader=None,
                 first_stage_ckpt_path=None,
                 discriminator_ckpt_path=None):
        super().__init__()

        self.config = config
        self.save_dir_path = save_dir_path
        self.monitoring_metrics = monitoring_metrics
        self.image_uploader = uploader
        self.automatic_optimization = False

        self.configure_models()
        self.configure_losses()
        self.set_transform()

        self.one_hot_encoder = OneHotEncoder(
            n_classes=self.config.model.vqmodel.dict_size + 1
        ).forward

        self.calc_NMSE = NMSE_metric()
        self.calc_SSIM = SSIM_metric()
        self.calc_PSNR = PSNR_metric()

        if first_stage_ckpt_path:
            self.load_first_stage_from_ckpt(first_stage_ckpt_path)

        if discriminator_ckpt_path:
            self.load_discriminator_from_ckpt(discriminator_ckpt_path)

    def load_first_stage_from_ckpt(self, path, load_only_enc=False, ignore_keys=[]):
        state_dict = torch.load(path, map_location='cpu')['state_dict']

        encoder_dict = {}
        decoder_dict = {}
        for k in state_dict.keys():
            if k.startswith('encoder'):
                encoder_dict[k[len('encoder.'):]] = state_dict[k]

            if k.startswith('decoder'):
                decoder_dict[k[len('decoder.'):]] = state_dict[k]

        self.encoder.load_state_dict(encoder_dict, strict=True)

        if not load_only_enc:
            self.decoder.load_state_dict(decoder_dict, strict=False)

        print('Restored first stage models from {}'.format(path))

    def load_discriminator_from_ckpt(self, path):
        state_dict = torch.load(path, map_location='cpu')['state_dict']

        dis_dict = {}
        for k in state_dict.keys():
            if k.startswith('dis'):
                dis_dict[k[len('dis.'):]] = state_dict[k]

        self.dis.load_state_dict(dis_dict, strict=True)

        print('Restored the discriminator from {}'.format(path))

    def train_dataloader(self):
        config = self.config.dataset
        return get_data_loader(
            mode='train',
            dataset_name=config.dataset_name,
            root_dir_path=config.root_dir_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            modality=config.modality,
            augmentations=config.augmentations,
            drop_last=True,
            window_width=getattr_else_none(config, 'window_width'),
            window_center=getattr_else_none(config, 'window_center'),
            window_scale=getattr_else_none(config, 'window_scale'),
        )

    def val_dataloader(self):
        config = self.config.dataset
        return get_data_loader(
            mode='val',
            dataset_name=config.dataset_name,
            root_dir_path=config.root_dir_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            modality=config.modality,
            augmentations=None,
            drop_last=False,
            window_width=getattr_else_none(config, 'window_width'),
            window_center=getattr_else_none(config, 'window_center'),
            window_scale=getattr_else_none(config, 'window_scale'),
        )

    def test_dataloader(self):
        config = self.config.dataset
        return get_data_loader(
            mode='test',
            dataset_name=config.dataset_name,
            root_dir_path=config.root_dir_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            modality=config.modality,
            augmentations=None,
            drop_last=False,
            window_width=getattr_else_none(config, 'window_width'),
            window_center=getattr_else_none(config, 'window_center'),
            window_scale=getattr_else_none(config, 'window_scale'),
        )

    def configure_optimizers(self):
        enc_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            self.config.enc_optim.lr, betas=(self.config.enc_optim.b1, self.config.enc_optim.b2),
            weight_decay=self.config.enc_optim.weight_decay,
        )

        dec_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.decoder.parameters()),
            self.config.dec_optim.lr, betas=(self.config.dec_optim.b1, self.config.dec_optim.b2),
            weight_decay=self.config.dec_optim.weight_decay,
        )

        dis_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()),
            self.config.dis_optim.lr, betas=(self.config.dis_optim.b1, self.config.dis_optim.b2),
            weight_decay=self.config.dis_optim.weight_decay,
        )

        return [enc_optim, dec_optim, dis_optim], []

    def training_epoch_end(self, training_step_outputs):
        if self.config.model.vqmodel.use_dropblock:
            self.decoder.dropblock.step()

    def configure_models(self):
        gen_config = self.config.model.vqmodel
        dis_config = self.config.model.dis

        self.encoder = UNetEncoder(
            in_channels=gen_config.in_channels,
            filters=gen_config.enc_filters,
            dict_size=gen_config.dict_size,
            momentum=gen_config.momentum,
            knn_backend=gen_config.knn_backend,
            use_styled_up_block=gen_config.enc_use_styled_up_block,
            num_gpus=self.config.run.num_gpus,
            init_embed=not gen_config.use_init_embed,
        )

        if hasattr(gen_config, 'model_name') and gen_config.model_name == 'VQGAN':
            vqgan_config = self.config.model.vqgan

            self.decoder = VQGAN(
                in_channels=vqgan_config.in_channels,
                mid_channels=vqgan_config.mid_channels,
                out_channels=vqgan_config.out_channels,
                emb_dim=vqgan_config.emb_dim,
                dict_size=vqgan_config.dict_size,
                enc_ch_multiplier=vqgan_config.enc_ch_multiplier,
                dec_ch_multiplier=vqgan_config.dec_ch_multiplier,
                num_res_blocks=vqgan_config.num_res_blocks,
                enc_attn_resolutions=vqgan_config.enc_attn_resolutions,
                dec_attn_resolutions=vqgan_config.dec_attn_resolutions,
                resolution=vqgan_config.resolution,
                p_dropout=vqgan_config.p_dropout,
                resamp_with_conv=vqgan_config.resamp_with_conv,
                knn_backend=vqgan_config.knn_backend,
            )

        else:
            self.decoder = UNetDecoder(
                in_channels=gen_config.enc_filters[0],
                out_channels=gen_config.in_channels,
                filters=gen_config.dec_filters,
                use_dropblock=gen_config.use_dropblock,
                block_size=gen_config.block_size,
                start_value=gen_config.start_value,
                stop_value=gen_config.stop_value,
                nr_steps=gen_config.nr_steps,
                dropped_skip_layers=gen_config.dropped_skip_layers,
                use_styled_up_block=gen_config.dec_use_styled_up_block,
                use_pixel_shuffle=gen_config.use_pixel_shuffle,
            )

        if dis_config.model_name == 'UNetDiscriminator':
            self.dis = UNetDiscriminator(
                in_channels=gen_config.in_channels,
                D_ch=dis_config.D_ch,
                D_wide=dis_config.D_wide,
                D_attn=dis_config.D_attn,
                resolution=dis_config.resolution,
                unconditional=True
            )

        elif dis_config.model_name == 'NLayerDiscriminator':
            self.dis = NLayerDiscriminator(
                in_channels=gen_config.in_channels,
                out_channels=1,
                n_filters=dis_config.n_filters,
                n_layers=dis_config.n_layers,
                normalization=dis_config.normalization,
            )

            if dis_config.apply_spectral_norm:
                apply_spectral_norm(self.dis)

    def configure_losses(self):
        config = self.config.loss

        self.embed_loss = EmbeddingLoss(
            dict_size=self.config.model.vqmodel.dict_size,
            margin=config.embed_loss.margin,
            use_distance_loss=config.embed_loss.use_distance_loss,
            use_regularization_loss=config.embed_loss.use_regularization_loss,
        )

        if config.use_perceptual_loss:
            if config.perceptual_loss_type == 'vgg':
                self.perceptual_loss = VGGLoss()
            elif config.perceptual_loss_type == 'lpips':
                self.perceptual_loss = LPIPSLoss()

        if config.use_frequency_loss:
            self.frequency_loss = FFL(loss_weight=1.0, alpha=1.0)

    def set_transform(self):
        self.transform_1 = RandomTransform(config=self.config.augmentation)
        self.transform_2 = RandomTransform(config=self.config.augmentation)

    def unet_perceptual_loss(self, output, target):
        loss_list = []
        for o, t in zip(output, target):
            loss_list.append(F.mse_loss(o, t.detach(), reduction='mean'))
        return torch.sum(torch.stack(loss_list))

    def to_lung(self, image, lung_window=LUNG_WINDOW):
        image = denormalize(image,
                            width=self.config.dataset.window_width,
                            center=self.config.dataset.window_center,
                            scale=self.config.dataset.window_scale)

        image = normalize(image.clone(),
                          width=lung_window['width'],
                          center=lung_window['center'],
                          scale=lung_window['scale'])

        return image

    def to_mediastinal(self, image, mediastinal_window=MEDIASTINAL_WINDOW):
        image = denormalize(image,
                            width=self.config.dataset.window_width,
                            center=self.config.dataset.window_center,
                            scale=self.config.dataset.window_scale)

        image = normalize(image.clone(),
                          width=mediastinal_window['width'],
                          center=mediastinal_window['center'],
                          scale=mediastinal_window['scale'])

        return image

    def denormalize_ct_values(self, image):
        return denormalize(
            image,
            width=self.config.dataset.window_width,
            center=self.config.dataset.window_center,
            scale=self.config.dataset.window_scale,
        )
