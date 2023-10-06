import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):

    def __init__(self, net='alex'):
        super().__init__()

        self.loss_func = lpips.LPIPS(net=net)

    def forward(self, sr, hr):
        b, n_channels, h, w = sr.size()
        sr = sr.expand(b, 3, h, w)
        hr = hr.expand(b, 3, h, w)
        loss = self.loss_func(sr, hr)
        return torch.mean(loss)
