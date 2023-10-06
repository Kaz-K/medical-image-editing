import os
import fsspec
import json
import logging
import numpy as np
import collections
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Dict
from typing import Tuple
from weakref import proxy

import torch
from torchvision.utils import save_image
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import rank_zero_only


log = logging.getLogger(__name__)

pathlike = Union[Path, str]


def get_filesystem(path: pathlike):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


class ModelSaver(ModelCheckpoint):

    def __init__(self, limit_num, save_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_num = limit_num
        self.save_interval = save_interval

    def save_checkpoint(self, trainer: "pl.Trainer") -> None:
        """Performs the main logic around saving a checkpoint.
        This method runs on all ranks. It is the responsibility of `trainer.save_checkpoint` to correctly handle the
        behaviour in distributed training, i.e., saving only on rank 0 for data parallel use cases.
        """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self._validate_monitor_key(trainer)

        # track epoch when ckpt was last checked
        self._last_global_step_saved = global_step

        # what can be monitored
        monitor_candidates = self._monitor_candidates(
            trainer, epoch=epoch, step=global_step)

        # callback supports multiple simultaneous modes
        # here we call each mode sequentially
        # Mode 1: save the top k checkpoints
        self._save_top_k_checkpoint(trainer, monitor_candidates)
        # Mode 2: save monitor=None checkpoints
        self._save_none_monitor_checkpoint(trainer, monitor_candidates)
        # Mode 3: save last checkpoints
        self._save_last_checkpoint(trainer, monitor_candidates)
        # Mode 4 (original): delete old checkpoints
        if trainer.is_global_zero and trainer.logger:
            self._delete_old_checkpoint(trainer)

        # notify loggers
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.after_save_checkpoint(proxy(self))

    def _delete_old_checkpoint(self, trainer):
        checkpoints = sorted(
            [c for c in os.listdir(self.dirpath) if 'ckpt-epoch' in c])
        if len(checkpoints) > self.limit_num:
            margin = len(checkpoints) - self.limit_num
            checkpoints_for_delete = checkpoints[:margin]

            for ckpt in checkpoints_for_delete:
                ckpt_epoch = int(
                    ckpt[len("ckpt-epoch="): len("ckpt-epoch=") + 4])
                if (ckpt_epoch + 1) % self.save_interval != 0:
                    model_path = os.path.join(self.dirpath, ckpt)
                    os.remove(model_path)


class Logger(LightningLoggerBase):

    def __init__(
        self,
        save_dir: str,
        config: collections.defaultdict,
        monitoring_metrics: list,
        uploader: 'ImageUploader' = None,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""
        self._config = config
        self._version = version
        self._fs = get_filesystem(save_dir)

        self._experiment = None
        self._monitoring_metrics = monitoring_metrics
        self._uploader = uploader
        self._kwargs = kwargs

    @property
    def root_dir(self) -> str:
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def log_dir(self) -> str:
        version = self.version if isinstance(
            self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        values = []
        for key in self._monitoring_metrics:
            if key in metrics.keys():
                v = metrics[key]
                if isinstance(v, torch.Tensor):
                    v = str(v.sum().item())
                else:
                    v = str(v)
            else:
                v = ''
            values.append(v)

        fname = os.path.join(self.log_dir, 'log.csv')
        os.makedirs(self.log_dir, exist_ok=True)
        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(self._monitoring_metrics), file=f)
            print(','.join(values), file=f)

        try:
            if self._uploader:
                self._uploader.send_image(
                    fname,
                    message='log',
                )
        except Exception as e:
            print("self._uploader.send_image error")

    @rank_zero_only
    def log_val_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'val_logs.csv')
        os.makedirs(self.log_dir, exist_ok=True)

        columns = metrics.keys()
        values = [str(value) for value in metrics.values()]

        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(columns), file=f)
            print(','.join(values), file=f)

    @rank_zero_only
    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'test_logs.csv')
        os.makedirs(self.log_dir, exist_ok=True)

        columns = metrics.keys()
        values = [str(value) for value in metrics.values()]

        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(columns), file=f)
            print(','.join(values), file=f)

        print('Test results are saved: {}'.format(fname))

    @rank_zero_only
    def log_hyperparams(self, seed_list):
        config_to_save = collections.defaultdict(dict)

        for key, child in self._config._asdict().items():
            for k, v in child._asdict().items():
                config_to_save[key][k] = v

        config_to_save['seed_list'] = seed_list
        config_to_save['save_dir_path'] = self.log_dir

        save_path = os.path.join(self.log_dir, 'config.json')
        os.makedirs(self.log_dir, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(config_to_save,
                      f,
                      ensure_ascii=False,
                      indent=2,
                      sort_keys=False,
                      separators=(',', ': '))

    @rank_zero_only
    def log_images(self, image_name: str, image: torch.Tensor, current_epoch: int, global_step: int, nrow: int) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        save_path = os.path.join(
            self.log_dir, f'{image_name}_{current_epoch:04d}_{global_step:06d}.png')
        os.makedirs(self.log_dir, exist_ok=True)
        save_image(image.data, save_path, nrow=nrow)

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = self
        return self._experiment

    @rank_zero_only
    def save(self) -> None:
        super().save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()
