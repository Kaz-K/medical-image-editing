import os
import random
import argparse
import warnings

import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import Callback

from trainers import MultiWindowTrainer
from trainers import SingleWindowTrainer
from trainers import VQGAN_UNetDis_Trainer
from utils import load_json
from utils import ModelSaver
from utils import Logger
from utils import InitSeedAndSaveConfig


warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)


load_dotenv()
TOKEN = os.environ.get("TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")


class ImageUploader(object):

    def __init__(self):
        self.client = WebClient(token=TOKEN)

    def send_image(self, file_path, message):
        try:
            result = self.client.files_upload(
                channels=CHANNEL_ID,
                initial_comment=message,
                file=file_path,
            )
            logger.info(result)

        except Exception as e:
            logger.error("Error uploading file: {}".format(e))


class SaveHyparamsCallback(Callback):

    def __init__(self, logger):
        self.logger = logger

    def on_init_start(self, trainer):
        self.logger.log_hyperparams()


def train_model(config, args):
    monitoring_metrics = config.run.monitoring_metrics

    logger = Logger(save_dir=config.save.save_dir,
                    config=config,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics,
                    uploader=ImageUploader())

    save_dir_path = logger.log_dir

    checkpoint_callback = ModelSaver(
        limit_num=10,
        save_interval=10,
        monitor=None,
        dirpath=save_dir_path,
        filename='ckpt-{epoch:04d}-{total_loss:.2f}',
        save_top_k=-1,
        save_last=False,
    )

    if args.multiwindow:
        Model = MultiWindowTrainer
    else:
        Model = SingleWindowTrainer

    if args.vqgan:
        Model = VQGAN_UNetDis_Trainer

    if config.run.resume_checkpoint:
        print('Loading model from {}'.format(config.run.resume_checkpoint))
        model = Model.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
            monitoring_metrics=monitoring_metrics,
            uploader=ImageUploader(),
            first_stage_ckpt_path=config.run.first_stage_ckpt_path,
            discriminator_ckpt_path=config.run.discriminator_ckpt_path,
        )

    else:
        model = Model(
            config=config,
            save_dir_path=save_dir_path,
            monitoring_metrics=monitoring_metrics,
            uploader=ImageUploader(),
            first_stage_ckpt_path=config.run.first_stage_ckpt_path,
            discriminator_ckpt_path=config.run.discriminator_ckpt_path,
        )

    trainer = pl.Trainer(gpus=config.run.visible_devices,
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=1,
                         accelerator='gpu' if config.run.distributed_backend == 'ddp' else None,
                         strategy=DDPPlugin(
                             find_unused_parameters=False) if config.run.distributed_backend == 'ddp' else config.run.distributed_backend,
                         deterministic=False,
                         logger=logger,
                         sync_batchnorm=True,
                         callbacks=[checkpoint_callback,
                                    InitSeedAndSaveConfig(logger, config)],
                         auto_lr_find=False,
                         num_sanity_val_steps=0 if not config.run.use_validation_sanity_check else 2,
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         limit_val_batches=2)

    return trainer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Editable medical image generation')
    parser.add_argument('-c', '--config', help='config', required=True)
    parser.add_argument('-m', '--mode', default='train', type=str)
    parser.add_argument('-w', '--multiwindow', action='store_true')
    parser.add_argument('-v', '--vqgan', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    seed = config.run.seed or random.randint(1, 10000)
    seed_everything(seed)

    print('Seed: {}'.format(seed))
    print('Config: ', config)

    trainer, model = train_model(config, args)

    if args.mode == 'train':
        trainer.fit(model)

    elif args.mode == 'test':
        trainer.test(model)
