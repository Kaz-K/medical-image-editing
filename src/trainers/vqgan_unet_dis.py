import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from focal_frequency_loss import FocalFrequencyLoss as FFL
from random import random
import functools

from .base import TrainerBase
from .base import SNAPSHOT_INTERVAL
from .base import CMAP
from networks import VQGAN
from networks import UNetDiscriminator
from functions import LPIPSLoss
from functions import hinge_d_loss
from functions import vanilla_d_loss
from utils import norm
from utils import denorm
from utils import to_cpu
from utils import apply_spectral_norm
from utils import to_image
from utils import subplot_image
from utils import cutmix_coordinates
from utils import cutmix
from utils import mask_src_tgt


class VQGAN_UNetDis_Trainer(TrainerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        w = self.config.loss.loss_weight

        _, dec_optim, dis_optim = self.optimizers()

        image = norm(denorm(batch['image'], vmin=0, vmax=1))
        recon, l_commit, _, _ = self.decoder(image)

        if self.config.loss.use_recon_loss:
            l_recon = F.mse_loss(recon, image, reduction='mean')
        else:
            l_recon = 0.0

        if self.config.loss.use_frequency_loss:
            l_frequency = self.frequency_loss(recon, image)
        else:
            l_frequency = 0.0 

        if self.config.loss.use_perceptual_loss:
            l_perceptual = self.perceptual_loss(recon, image)
        else:
            l_perceptual = 0.0 

        f_map, f_bottle, f_perceptual = self.dis(recon)

        l_gen = - (torch.mean(f_map) + torch.mean(f_bottle))

        if self.config.loss.use_unet_perceptual_loss:
            _, _, r_perceptual = self.dis(image.detach())
            l_unet_perceptual = self.unet_perceptual_loss(f_perceptual, r_perceptual)

        else:
            l_unet_perceptual = 0.0

        l_gen_total = w.recon * l_recon \
                    + w.freq * l_frequency \
                    + w.perceptual * l_perceptual \
                    + w.commit * l_commit \
                    + w.gen * l_gen \
                    + w.unet_perceptual * l_unet_perceptual

        dec_optim.zero_grad()
        self.manual_backward(l_gen_total)
        dec_optim.step()

        for _ in range(self.config.loss.n_inner_loops):
            r_map, r_bottle, _ = self.dis(image.detach())
            f_map, f_bottle, _ = self.dis(recon.detach())

            assert self.config.loss.dis_loss_type == 'hinge_d_loss'

            l_dis = hinge_d_loss(r_map, f_map) \
                  + hinge_d_loss(r_bottle, f_bottle)

            image_size = self.config.dataset.image_size

            mask = cutmix(
                torch.ones_like(r_map), torch.zeros_like(r_map),
                cutmix_coordinates(image_size[0], image_size[1])
            )

            if random() > 0.5:
                mask = 1 - mask

            cutmix_images = mask_src_tgt(image, recon, mask)

            cutmix_map, cutmix_bottle, _ = self.dis(cutmix_images.detach())

            cutmix_enc_loss = torch.mean(F.relu(1. + cutmix_bottle))
            cutmix_dec_loss = torch.mean(F.relu(1. - (mask * 2 - 1) * cutmix_map))

            l_cutmix = cutmix_enc_loss + cutmix_dec_loss

            rf_map = mask_src_tgt(r_map, f_map, mask)

            l_consistency = F.mse_loss(cutmix_map, rf_map)

            l_dis_total = w.dis * l_dis \
                        + w.cutmix * l_cutmix \
                        + w.consistency * l_consistency

            dis_optim.zero_grad()
            self.manual_backward(l_dis_total)
            dis_optim.step()

        l_total = l_gen_total + l_dis_total

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total', l_total, prog_bar=True)
        self.log('gen_total', l_gen_total, prog_bar=True)
        self.log('recon', w.recon * l_recon, prog_bar=True)
        self.log('freq', w.freq * l_frequency, prog_bar=True)
        self.log('perceptual', w.perceptual * l_perceptual, prog_bar=True)
        self.log('commit', w.commit * l_commit, prog_bar=True)
        self.log('gen', w.gen * l_gen, prog_bar=True)
        self.log('unet_perceptual', w.unet_perceptual * l_unet_perceptual, prog_bar=True)
        self.log('dis_total', l_dis_total, prog_bar=True)
        self.log('dis', w.dis * l_dis, prog_bar=True)
        self.log('cutmix', w.cutmix * l_cutmix, prog_bar=True)
        self.log('consistency', w.consistency * l_consistency, prog_bar=True)

        if self.global_step % SNAPSHOT_INTERVAL == 0 and self.trainer.is_global_zero:
            dict_size = self.config.model.vqmodel.dict_size

            image = to_image(image)
            recon = to_image(recon)

            r_map = to_image(r_map)
            f_map = to_image(f_map)
            cutmix_images = to_image(cutmix_images)
            cutmix_map = to_image(cutmix_map)
            rf_map = to_image(rf_map)
            mask = to_image(mask)

            n_col = 6
            n_row = 2
            _subplot_image = functools.partial(subplot_image, x=n_row, y=n_col, fontsize=3)

            i = 0
            _subplot_image(image, 'image', 'gray', -1, 1, z=i * n_col + 1)
            _subplot_image(recon, 'recon', 'gray', -1, 1, z=i * n_col + 2)

            i = 1
            _subplot_image(r_map, 'r_map', 'gray', None, None, z=i * n_col + 1)
            _subplot_image(f_map, 'f_map', 'gray', None, None, z=i * n_col + 2)
            _subplot_image(cutmix_images, 'cutmix_images', 'gray', -1, 1, z=i * n_col + 3)
            _subplot_image(cutmix_map, 'cutmix_map', 'gray', None, None, z=i * n_col + 4)
            _subplot_image(rf_map, 'rf_map', 'gray', None, None, z=i * n_col + 5)
            _subplot_image(mask, 'mask', 'gray', None, None, z=i * n_col + 6)


            image_save_path = os.path.join(
                self.save_dir_path, 'train_' + str(self.global_step).zfill(6) + '.png'
            )

            plt.savefig(image_save_path, bbox_inches='tight', dpi=512)
            plt.clf()

            message = """
            Global Step: {}
            """.format(
                self.global_step
            )

            self.image_uploader.send_image(
                image_save_path, message=message,
            )

        return l_total

    def validation_step(self, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return

        with torch.no_grad():
            image = norm(denorm(batch['image'], vmin=0, vmax=1))
            recon, _, _, _ = self.decoder(image)

            r_map, _, _ = self.dis(image.detach())
            f_map, _, _ = self.dis(recon.detach())

        batch_size = image.size(0)

        if self.config.dataset.dataset_name == 'CRCDataset':
            image = to_image(image, retain_batch=True)
            recon = to_image(recon, retain_batch=True)

            r_map = to_image(r_map, retain_batch=True)
            f_map = to_image(f_map, retain_batch=True)

            n_save_images = self.config.save.n_save_images
            n_rows = min(n_save_images, batch_size)
            dict_size = self.config.model.vqmodel.dict_size

            n_cols = 4
            for i in range(n_rows):
                img = image[i, ...]
                rec = recon[i, ...]
                r_m = r_map[i, ...]
                f_m = f_map[i, ...]

                subplot_image(img, 'l_img', 'gray', -1, 1, n_rows, n_cols, n_cols * i + 1)
                subplot_image(rec, 'l_rec', 'gray', -1, 1, n_rows, n_cols, n_cols * i + 2)
                subplot_image(r_m, 'r_m', 'gray', None, None, n_rows, n_cols, n_cols * i + 3)
                subplot_image(f_m, 'f_m', 'gray', None, None, n_rows, n_cols, n_cols * i + 4)

        else:
            raise NotImplementedError

        plt.savefig(os.path.join(
            self.save_dir_path, str(self.current_epoch).zfill(6) + '.png'
        ), bbox_inches='tight', dpi=512)
        plt.clf()

        self.image_uploader.send_image(
            os.path.join(
                self.save_dir_path, str(self.current_epoch).zfill(6) + '.png'
            ), message=self.global_step,
        )

        if self.current_epoch == 0:
            if self.config.dataset.dataset_name == 'CRCDataset':
                l_img = img 
                m_img = img 

            plt.imshow(l_img, cmap='gray', vmin=-1, vmax=1)
            plt.savefig(os.path.join(
                self.save_dir_path, 'l_img_' + str(self.current_epoch).zfill(6) + '.png'
            ), bbox_inches='tight', dpi=512)
            plt.clf()

            self.image_uploader.send_image(
                os.path.join(
                    self.save_dir_path, 'l_img_' + str(self.current_epoch).zfill(6) + '.png'
                ), message=self.global_step,
            )

            plt.imshow(m_img, cmap='gray', vmin=-1, vmax=1)
            plt.savefig(os.path.join(
                self.save_dir_path, 'm_img_' + str(self.current_epoch).zfill(6) + '.png'
            ), bbox_inches='tight', dpi=512)
            plt.clf()

            self.image_uploader.send_image(
                os.path.join(
                    self.save_dir_path, 'm_img_' + str(self.current_epoch).zfill(6) + '.png'
                ), message=self.global_step,
            )

        if self.config.dataset.dataset_name == 'CRCDataset':
            l_rec = rec 
            m_rec = rec 

        plt.imshow(l_rec, cmap='gray', vmin=-1, vmax=1)
        plt.savefig(os.path.join(
            self.save_dir_path, 'l_rec_' + str(self.current_epoch).zfill(6) + '.png'
        ), bbox_inches='tight', dpi=512)
        plt.clf()

        self.image_uploader.send_image(
            os.path.join(
                self.save_dir_path, 'l_rec_' + str(self.current_epoch).zfill(6) + '.png'
            ), message=self.global_step,
        )

        plt.imshow(m_rec, cmap='gray', vmin=-1, vmax=1)
        plt.savefig(os.path.join(
            self.save_dir_path, 'm_rec_' + str(self.current_epoch).zfill(6) + '.png'
        ), bbox_inches='tight', dpi=512)
        plt.clf()

        self.image_uploader.send_image(
            os.path.join(
                self.save_dir_path, 'm_rec_' + str(self.current_epoch).zfill(6) + '.png'
            ), message=self.global_step,
        )
