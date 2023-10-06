import random

import torch 
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import seed_everything


class InitSeedAndSaveConfig(Callback):

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def setup(self, trainer, pl_module, stage=None):
        # init seed
        rank = pl_module.global_rank

        if self.config.run.seed_list:
            seed = self.config.run.seed_list[rank]
        else:
            seed = random.randint(1, 10000)

        seed_everything(seed)
        print('Seed set to {} in gpu-rank: {}'.format(seed, rank))

        # save config with seed list
        num_gpus = self.config.run.num_gpus
        seed_on_gpu = torch.Tensor([seed]).int().to(torch.device('cuda', rank))
        seed_list = [torch.zeros_like(seed_on_gpu) for _ in range(num_gpus)]
        torch.distributed.all_gather(seed_list, seed_on_gpu)

        if pl_module.trainer.is_global_zero:
            seed_list = [s.item() for s in seed_list]

        self.logger.log_hyperparams(seed_list)
