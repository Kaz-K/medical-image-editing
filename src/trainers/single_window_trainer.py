import os
import numpy as np
import pandas as pd
import nibabel as nib
import functools
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia as K
import matplotlib.pyplot as plt
import seaborn as sns
from focal_frequency_loss import FocalFrequencyLoss as FFL
from scipy.stats import entropy

from .base import TrainerBase
from .base import LUNG_WINDOW
from .base import MEDIASTINAL_WINDOW
from .base import CMAP
from .base import SNAPSHOT_INTERVAL
from functions import hinge_d_loss
from utils import mask_src_tgt
from utils import subplot_image
from utils import cutmix_coordinates
from utils import cutmix
from utils import to_image
from utils import save_image
from utils import save_fused_image
from utils import norm
from utils import denorm
from utils import t_normalize as normalize
from utils import denormalize
from utils import to_nifti


class SingleWindowTrainer(TrainerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.config.run.training_mode == 'first_step':
            return self._train_first_step(batch, batch_idx)

        elif self.config.run.training_mode == 'second_step':
            if self.config.model.dis.model_name == 'UNetDiscriminator':
                return self._train_second_step_unet_dis(batch, batch_idx)

            elif self.config.model.dis.model_name == 'NLayerDiscriminator':
                return self._train_second_step_nl_dis(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.config.run.training_mode == 'inference':
            return self._inference(batch, batch_idx)

        else:
            return self._test_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        if self.config.run.training_mode == 'inference':
            pass

        else:
            return self._test_epoch_end(outputs)

    def _train_first_step(self, batch, batch_idx):
        w = self.config.loss.loss_weight

        enc_optim, dec_optim, _ = self.optimizers()

        image = denorm(batch['image'], vmin=0, vmax=1)

        noised_image_1, clear_image_1 = self.transform_1(image)
        noised_image_2, clear_image_2 = self.transform_2(image)

        noised_image_1 = norm(noised_image_1)
        noised_image_2 = norm(noised_image_2)

        clear_image_1 = norm(clear_image_1)
        clear_image_2 = norm(clear_image_2)

        embed_1, l_commit_1, ids_1 = self.encoder(
            noised_image_1, rank=self.global_rank)
        embed_2, l_commit_2, ids_2 = self.encoder(
            noised_image_2, rank=self.global_rank)

        l_commit = l_commit_1 + l_commit_2

        r_ids_1 = self.transform_2.forward_transform(
            self.transform_1.reverse_transform(ids_1)
        ).int()
        r_ids_2 = self.transform_1.forward_transform(
            self.transform_2.reverse_transform(ids_2)
        ).int()

        r_ids_1 = self.one_hot_encoder(r_ids_1)[:, 1:, ...]
        r_ids_2 = self.one_hot_encoder(r_ids_2)[:, 1:, ...]

        codebook = self.encoder.vq.get_codebook()

        l_cross, l_dist, l_reg = self.embed_loss(embed_1, r_ids_1,
                                                 embed_2, r_ids_2,
                                                 codebook)

        recon_1 = self.decoder(embed_1)
        recon_2 = self.decoder(embed_2)

        if self.config.loss.use_recon_loss:
            l_rec_1 = F.mse_loss(recon_1, clear_image_1, reduction='mean')
            l_rec_2 = F.mse_loss(recon_2, clear_image_2, reduction='mean')
            l_recon = l_rec_1 + l_rec_2
        else:
            l_recon = 0.0

        if self.config.loss.use_frequency_loss:
            l_freq_1 = self.frequency_loss(recon_1, clear_image_1)
            l_freq_2 = self.frequency_loss(recon_2, clear_image_2)
            l_frequency = l_freq_1 + l_freq_2
        else:
            l_frequency = 0.0

        if self.config.loss.use_perceptual_loss:
            l_percep_1 = self.perceptual_loss(recon_1, clear_image_1)
            l_percep_2 = self.perceptual_loss(recon_2, clear_image_2)
            l_perceptual = l_percep_1 + l_percep_2
        else:
            l_perceptual = 0.0

        l_gen_total = w.commit * l_commit \
            + w.cross * l_cross \
            + w.dist * l_dist \
            + w.reg * l_reg \
            + w.recon * l_recon \
            + w.freq * l_frequency \
            + w.perceptual * l_perceptual

        enc_optim.zero_grad()
        dec_optim.zero_grad()

        self.manual_backward(l_gen_total)

        enc_optim.step()
        dec_optim.step()

        l_total = l_gen_total

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total', l_total, prog_bar=True)
        self.log('gen_total', l_gen_total, prog_bar=True)
        self.log('commit', w.commit * l_commit, prog_bar=True)
        self.log('cross', w.cross * l_cross, prog_bar=True)
        self.log('dist', w.dist * l_dist, prog_bar=True)
        self.log('reg', w.reg * l_reg, prog_bar=True)
        self.log('recon', w.recon * l_recon, prog_bar=True)
        self.log('freq', w.freq * l_frequency, prog_bar=True)
        self.log('perceptual', w.perceptual * l_perceptual, prog_bar=True)

        if self.global_step % SNAPSHOT_INTERVAL == 0 and self.trainer.is_global_zero:
            dict_size = self.config.model.vqmodel.dict_size

            if self.config.dataset.dataset_name == 'CRCDataset':
                clear_image_1 = to_image(clear_image_1)
                recon_1 = to_image(recon_1)
                ids_1 = to_image(ids_1, is_ids=True)

                clear_image_2 = to_image(clear_image_2)
                recon_2 = to_image(recon_2)
                ids_2 = to_image(ids_2, is_ids=True)

                print('IDs-1: ', np.bincount(ids_1.ravel()))
                print('IDs-2: ', np.bincount(ids_2.ravel()))

                n_col = 3
                n_row = 2
                _subplot_image = functools.partial(
                    subplot_image, x=n_row, y=n_col, fontsize=3)

                i = 0
                _subplot_image(clear_image_1, 'clear_image_1',
                               'gray', -1, 1, z=i * n_col + 1)
                _subplot_image(recon_1, 'recon_1', 'gray', -
                               1, 1, z=i * n_col + 2)
                _subplot_image(ids_1, 'ids_1', CMAP, 0,
                               dict_size, z=i * n_col + 3)

                i = 1
                _subplot_image(clear_image_2, 'clear_image_2',
                               'gray', -1, 1, z=i * n_col + 1)
                _subplot_image(recon_2, 'recon_2', 'gray', -
                               1, 1, z=i * n_col + 2)
                _subplot_image(ids_2, 'ids_2', CMAP, 0,
                               dict_size, z=i * n_col + 3)

            else:
                l_clear_image_1 = to_image(self.to_lung(clear_image_1))
                m_clear_image_1 = to_image(self.to_mediastinal(clear_image_1))
                l_recon_1 = to_image(self.to_lung(recon_1))
                m_recon_1 = to_image(self.to_mediastinal(recon_1))
                ids_1 = to_image(ids_1, is_ids=True)

                l_clear_image_2 = to_image(self.to_lung(clear_image_2))
                m_clear_image_2 = to_image(self.to_mediastinal(clear_image_2))
                l_recon_2 = to_image(self.to_lung(recon_2))
                m_recon_2 = to_image(self.to_mediastinal(recon_2))
                ids_2 = to_image(ids_2, is_ids=True)

                print('IDs-1: ', np.bincount(ids_1.ravel()))
                print('IDs-2: ', np.bincount(ids_2.ravel()))

                n_col = 5
                n_row = 2
                _subplot_image = functools.partial(
                    subplot_image, x=n_row, y=n_col, fontsize=3)

                i = 0
                _subplot_image(l_clear_image_1, 'l_clear_image_1',
                               'gray', -1, 1, z=i * n_col + 1)
                _subplot_image(m_clear_image_1, 'm_clear_image_1',
                               'gray', -1, 1, z=i * n_col + 2)
                _subplot_image(l_recon_1, 'l_recon_1',
                               'gray', -1, 1, z=i * n_col + 3)
                _subplot_image(m_recon_1, 'm_recon_1',
                               'gray', -1, 1, z=i * n_col + 4)
                _subplot_image(ids_1, 'ids_1', CMAP, 0,
                               dict_size, z=i * n_col + 5)

                i = 1
                _subplot_image(l_clear_image_2, 'l_clear_image_2',
                               'gray', -1, 1, z=i * n_col + 1)
                _subplot_image(m_clear_image_2, 'm_clear_image_2',
                               'gray', -1, 1, z=i * n_col + 2)
                _subplot_image(l_recon_2, 'l_recon_2',
                               'gray', -1, 1, z=i * n_col + 3)
                _subplot_image(m_recon_2, 'm_recon_2',
                               'gray', -1, 1, z=i * n_col + 4)
                _subplot_image(ids_2, 'ids_2', CMAP, 0,
                               dict_size, z=i * n_col + 5)

            image_save_path = os.path.join(
                self.save_dir_path, 'train_' +
                str(self.global_step).zfill(6) + '.png'
            )

            plt.savefig(image_save_path, bbox_inches='tight', dpi=512)
            plt.clf()

            message = """
            Global Step: {}, IDs-1: {}, IDs-2: {}
            """.format(
                self.global_step,
                ' '.join([str(s) for s in list(np.bincount(ids_1.ravel()))]),
                ' '.join([str(s) for s in list(np.bincount(ids_2.ravel()))])
            )

            self.image_uploader.send_image(
                image_save_path, message=message,
            )

        return l_total

    def _train_second_step_unet_dis(self, batch, batch_idx):
        self.encoder.eval()

        w = self.config.loss.loss_weight

        _, dec_optim, dis_optim = self.optimizers()

        image = norm(denorm(batch['image'], vmin=0, vmax=1))

        with torch.no_grad():
            embed, _, ids = self.encoder(image, rank=self.global_rank)

        recon = self.decoder(embed.detach())

        if self.config.loss.use_recon_loss:
            if self.config.loss.use_l1_loss:
                l_recon = F.l1_loss(recon, image, reduction='mean')
            else:
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
            l_unet_perceptual = self.unet_perceptual_loss(
                f_perceptual, r_perceptual)

        else:
            l_unet_perceptual = 0.0

        l_gen_total = w.recon * l_recon \
            + w.freq * l_frequency \
            + w.perceptual * l_perceptual \
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
            cutmix_dec_loss = torch.mean(
                F.relu(1. - (mask * 2 - 1) * cutmix_map))

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
        self.log('gen', w.gen * l_gen, prog_bar=True)
        self.log('unet_perceptual', w.unet_perceptual *
                 l_unet_perceptual, prog_bar=True)
        self.log('dis_total', l_dis_total, prog_bar=True)
        self.log('dis', w.dis * l_dis, prog_bar=True)
        self.log('cutmix', w.cutmix * l_cutmix, prog_bar=True)
        self.log('consistency', w.consistency * l_consistency, prog_bar=True)

        if self.global_step % SNAPSHOT_INTERVAL == 0 and self.trainer.is_global_zero:
            dict_size = self.config.model.vqmodel.dict_size

            image = to_image(image)
            recon = to_image(recon)
            ids = to_image(ids, is_ids=True)

            r_map = to_image(r_map)
            f_map = to_image(f_map)
            cutmix_images = to_image(cutmix_images)
            cutmix_map = to_image(cutmix_map)
            rf_map = to_image(rf_map)
            mask = to_image(mask)

            print('IDs: ', np.bincount(ids.ravel()))

            n_col = 6
            n_row = 2
            _subplot_image = functools.partial(
                subplot_image, x=n_row, y=n_col, fontsize=3)

            i = 0
            _subplot_image(image, 'image', 'gray', -1, 1, z=i * n_col + 1)
            _subplot_image(recon, 'recon', 'gray', -1, 1, z=i * n_col + 3)
            _subplot_image(ids, 'ids_1', CMAP, 0, dict_size, z=i * n_col + 5)

            i = 1
            _subplot_image(r_map, 'r_map', 'gray', None, None, z=i * n_col + 1)
            _subplot_image(f_map, 'f_map', 'gray', None, None, z=i * n_col + 2)
            _subplot_image(cutmix_images, 'cutmix_images',
                           'gray', -1, 1, z=i * n_col + 3)
            _subplot_image(cutmix_map, 'cutmix_map', 'gray',
                           None, None, z=i * n_col + 4)
            _subplot_image(rf_map, 'rf_map', 'gray',
                           None, None, z=i * n_col + 5)
            _subplot_image(mask, 'mask', 'gray', None, None, z=i * n_col + 6)

            image_save_path = os.path.join(
                self.save_dir_path, 'train_' +
                str(self.global_step).zfill(6) + '.png'
            )

            plt.savefig(image_save_path, bbox_inches='tight', dpi=512)
            plt.clf()

            message = """
            Global Step: {}, IDs: {}
            """.format(
                self.global_step,
                ' '.join([str(s) for s in list(np.bincount(ids.ravel()))])
            )

            self.image_uploader.send_image(
                image_save_path, message=message,
            )

        return l_total

    def _train_second_step_nl_dis(self, batch, batch_idx):
        self.encoder.eval()

        w = self.config.loss.loss_weight

        _, dec_optim, dis_optim = self.optimizers()

        image = norm(denorm(batch['image'], vmin=0, vmax=1))

        with torch.no_grad():
            embed, _, ids = self.encoder(image, rank=self.global_rank)

        recon = self.decoder(embed.detach())

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

        l_gen = - torch.mean(self.dis(recon))

        l_gen_total = w.recon * l_recon \
            + w.freq * l_frequency \
            + w.perceptual * l_perceptual \
            + w.gen * l_gen

        dec_optim.zero_grad()
        self.manual_backward(l_gen_total)
        dec_optim.step()

        for _ in range(self.config.loss.n_inner_loops):
            l_real = self.dis(image.detach())
            l_fake = self.dis(recon.detach())

            assert self.config.loss.dis_loss_type == 'hinge_d_loss'

            l_dis = hinge_d_loss(l_real, l_fake)

            l_dis_total = w.dis * l_dis

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
        self.log('gen', w.gen * l_gen, prog_bar=True)
        self.log('dis_total', l_dis_total, prog_bar=True)
        self.log('dis', w.dis * l_dis, prog_bar=True)

        if self.global_step % SNAPSHOT_INTERVAL == 0 and self.trainer.is_global_zero:
            dict_size = self.config.model.vqmodel.dict_size

            image = to_image(image)
            recon = to_image(recon)
            ids = to_image(ids, is_ids=True)

            print('IDs: ', np.bincount(ids.ravel()))

            n_col = 3
            n_row = 1
            _subplot_image = functools.partial(
                subplot_image, x=n_row, y=n_col, fontsize=3)

            i = 0
            _subplot_image(image, 'image', 'gray', -1, 1, z=i * n_col + 1)
            _subplot_image(recon, 'recon', 'gray', -1, 1, z=i * n_col + 2)
            _subplot_image(ids, 'ids_1', CMAP, 0, dict_size, z=i * n_col + 3)

            image_save_path = os.path.join(
                self.save_dir_path, 'train_' +
                str(self.global_step).zfill(6) + '.png'
            )

            plt.savefig(image_save_path, bbox_inches='tight', dpi=512)
            plt.clf()

            message = """
            Global Step: {}, IDs: {}
            """.format(
                self.global_step,
                ' '.join([str(s) for s in list(np.bincount(ids.ravel()))])
            )

            self.image_uploader.send_image(
                image_save_path, message=message,
            )

        return l_total

    def validation_step(self, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return

        image = batch['image']
        batch_size = image.size(0)

        with torch.no_grad():
            embed, _, ids = self.encoder(image, rank=self.global_rank)
            recon = self.decoder(embed)

        if self.config.run.training_mode == 'first_step':
            r_map = f_map = torch.zeros_like(image)

        elif self.config.run.training_mode == 'second_step':
            if self.config.model.dis.model_name == 'UNetDiscriminator':
                r_map, _, _ = self.dis(image.detach())
                f_map, _, _ = self.dis(recon.detach())

            elif self.config.model.dis.model_name == 'NLayerDiscriminator':
                r_map = f_map = torch.zeros_like(image)

        if self.config.dataset.dataset_name == 'CRCDataset':
            image = to_image(image, retain_batch=True)
            recon = to_image(recon, retain_batch=True)

            ids = to_image(ids, is_ids=True, retain_batch=True)

            r_map = to_image(r_map, retain_batch=True)
            f_map = to_image(f_map, retain_batch=True)

            print('IDs: ', np.bincount(ids.ravel()))

            n_save_images = self.config.save.n_save_images
            n_rows = min(n_save_images, batch_size)
            dict_size = self.config.model.vqmodel.dict_size

            n_cols = 7
            for i in range(n_rows):
                img = image[i, ...]
                rec = recon[i, ...]
                i_ids = ids[i, ...]
                r_m = r_map[i, ...]
                f_m = f_map[i, ...]

                subplot_image(img, 'l_img', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 1)
                subplot_image(rec, 'l_rec', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 2)
                subplot_image(i_ids, 'ids', CMAP, 0, dict_size,
                              n_rows, n_cols, n_cols * i + 3)
                subplot_image(r_m, 'r_m', 'gray', None, None,
                              n_rows, n_cols, n_cols * i + 4)
                subplot_image(f_m, 'f_m', 'gray', None, None,
                              n_rows, n_cols, n_cols * i + 5)

        else:
            l_image = to_image(self.to_lung(image), retain_batch=True)
            m_image = to_image(self.to_mediastinal(image), retain_batch=True)

            l_recon = to_image(self.to_lung(recon), retain_batch=True)
            m_recon = to_image(self.to_mediastinal(recon), retain_batch=True)

            ids = to_image(ids, is_ids=True, retain_batch=True)

            r_map = to_image(r_map, retain_batch=True)
            f_map = to_image(f_map, retain_batch=True)

            print('IDs: ', np.bincount(ids.ravel()))

            n_save_images = self.config.save.n_save_images
            n_rows = min(n_save_images, batch_size)
            dict_size = self.config.model.vqmodel.dict_size

            n_cols = 7
            for i in range(n_rows):
                l_img = l_image[i, ...]
                l_rec = l_recon[i, ...]
                m_img = m_image[i, ...]
                m_rec = m_recon[i, ...]
                i_ids = ids[i, ...]
                r_m = r_map[i, ...]
                f_m = f_map[i, ...]

                subplot_image(l_img, 'l_img', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 1)
                subplot_image(l_rec, 'l_rec', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 2)
                subplot_image(m_img, 'm_img', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 3)
                subplot_image(m_rec, 'm_rec', 'gray', -1, 1,
                              n_rows, n_cols, n_cols * i + 4)
                subplot_image(i_ids, 'ids', CMAP, 0, dict_size,
                              n_rows, n_cols, n_cols * i + 5)
                subplot_image(r_m, 'r_m', 'gray', None, None,
                              n_rows, n_cols, n_cols * i + 6)
                subplot_image(f_m, 'f_m', 'gray', None, None,
                              n_rows, n_cols, n_cols * i + 7)

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
                self.save_dir_path, 'l_img_' +
                str(self.current_epoch).zfill(6) + '.png'
            ), bbox_inches='tight', dpi=512)
            plt.clf()

            self.image_uploader.send_image(
                os.path.join(
                    self.save_dir_path, 'l_img_' +
                    str(self.current_epoch).zfill(6) + '.png'
                ), message=self.global_step,
            )

            plt.imshow(m_img, cmap='gray', vmin=-1, vmax=1)
            plt.savefig(os.path.join(
                self.save_dir_path, 'm_img_' +
                str(self.current_epoch).zfill(6) + '.png'
            ), bbox_inches='tight', dpi=512)
            plt.clf()

            self.image_uploader.send_image(
                os.path.join(
                    self.save_dir_path, 'm_img_' +
                    str(self.current_epoch).zfill(6) + '.png'
                ), message=self.global_step,
            )

        if self.config.dataset.dataset_name == 'CRCDataset':
            l_rec = rec
            m_rec = rec

        plt.imshow(l_rec, cmap='gray', vmin=-1, vmax=1)
        plt.savefig(os.path.join(
            self.save_dir_path, 'l_rec_' +
            str(self.current_epoch).zfill(6) + '.png'
        ), bbox_inches='tight', dpi=512)
        plt.clf()

        self.image_uploader.send_image(
            os.path.join(
                self.save_dir_path, 'l_rec_' +
                str(self.current_epoch).zfill(6) + '.png'
            ), message=self.global_step,
        )

        plt.imshow(m_rec, cmap='gray', vmin=-1, vmax=1)
        plt.savefig(os.path.join(
            self.save_dir_path, 'm_rec_' +
            str(self.current_epoch).zfill(6) + '.png'
        ), bbox_inches='tight', dpi=512)
        plt.clf()

        self.image_uploader.send_image(
            os.path.join(
                self.save_dir_path, 'm_rec_' +
                str(self.current_epoch).zfill(6) + '.png'
            ), message=self.global_step,
        )

    def _inference(self, batch, batch_idx):
        image = batch['image']
        batch_size = image.size(0)
        patient_ids = batch['patient_id']
        slice_nums = batch['slice_num']

        with torch.no_grad():
            embed, _, ids = self.encoder(image, rank=self.global_rank)
            recon = self.decoder(embed)

        if self.config.dataset.dataset_name == 'NCCLungDataset':
            image = self.to_lung(image)
            recon = self.to_lung(recon)

        for i in range(image.size(0)):
            patient_id = patient_ids[i]
            slice_num = slice_nums[i].item()
            img = image[i, 0, ...].detach().cpu().numpy()
            rec = recon[i, 0, ...].detach().cpu().numpy()
            map = ids[i, ...].detach().cpu().numpy().astype(np.int32)

            if self.config.dataset.dataset_name == 'CRCDataset':
                img = np.flipud(img)
                rec = np.flipud(rec)
                map = np.flipud(map)

            save_dir_path = os.path.join(
                self.config.save.save_dir,
                self.config.save.study_name,
                patient_id
            )

            os.makedirs(save_dir_path, exist_ok=True)

            save_image(img, 'gray', -1, 1, os.path.join(
                save_dir_path, 'image_{}.png'.format(str(slice_num).zfill(4))
            ))

            save_image(rec, 'gray', -1, 1, os.path.join(
                save_dir_path, 'recon_{}.png'.format(str(slice_num).zfill(4))
            ))

            save_image(map, CMAP, 0, self.config.model.vqmodel.dict_size, os.path.join(
                save_dir_path, 'label_{}.png'.format(str(slice_num).zfill(4))
            ))

            img = to_nifti(img)
            rec = to_nifti(rec)
            map = to_nifti(map)

            nib.save(img, os.path.join(
                save_dir_path, 'image_{}.nii.gz'.format(
                    str(slice_num).zfill(4))
            ))

            nib.save(rec, os.path.join(
                save_dir_path, 'recon_{}.nii.gz'.format(
                    str(slice_num).zfill(4))
            ))

            nib.save(map, os.path.join(
                save_dir_path, 'label_{}.nii.gz'.format(
                    str(slice_num).zfill(4))
            ))

    def _test_step(self, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return

        image = batch['image']
        batch_size = image.size(0)

        with torch.no_grad():
            embed, _, ids = self.encoder(image, rank=self.global_rank)
            recon = self.decoder(embed)

        NMSE = self.calc_NMSE(recon, image)
        SSIM = self.calc_SSIM(recon, image)
        PSNR = self.calc_PSNR(recon, image)

        ids = ids.cpu().numpy()
        bin_counts = np.bincount(
            ids.ravel(), minlength=self.config.model.vqmodel.dict_size + 1)[1:]
        label_entropy = entropy(bin_counts, base=2)

        s_batch_idx = str(batch_idx).zfill(4)

        if batch_idx % 1 == 0:
            if self.config.dataset.dataset_name == 'CRCDataset':
                image = image[0, 0, ...].cpu().numpy()
                recon = recon[0, 0, ...].cpu().numpy()
                ids = ids[0, ...]

                image = np.flipud(image)
                recon = np.flipud(recon)
                ids = np.flipud(ids)

                save_image(image, cmap='gray', vmin=-1, vmax=1,
                           path=os.path.join(self.save_dir_path, 'image_{}.png'.format(s_batch_idx)))
                save_image(recon, cmap='gray', vmin=-1, vmax=1,
                           path=os.path.join(self.save_dir_path, 'recon_{}.png'.format(s_batch_idx)))
                save_image(ids, cmap=CMAP, vmin=0, vmax=self.config.model.vqmodel.dict_size,
                           path=os.path.join(self.save_dir_path, 'idx_{}.png'.format(s_batch_idx)))
                save_fused_image(recon, 'gray', -1, 1, ids, CMAP, 0, self.config.model.vqmodel.dict_size, 0.3,
                                 path=os.path.join(self.save_dir_path, 'fused_{}.png'.format(s_batch_idx)))

            else:
                raise NotImplementedError()

        return {
            'NMSE': NMSE.item(),
            'SSIM': SSIM.item(),
            'PSNR': PSNR.item(),
            'Entropy': label_entropy,
        }

    def _test_epoch_end(self, outputs):
        if not self.trainer.is_global_zero:
            return

        result = {}
        for key in outputs[0].keys():
            values = [o[key] for o in outputs]
            avg_value = np.mean(values)
            std_value = np.std(values)

            result.update({
                key + '_avg': [avg_value],
                key + '_std': [std_value],
            })

        df = pd.DataFrame.from_dict(result)
        df.to_csv(os.path.join(self.save_dir_path, 'result.csv'))
