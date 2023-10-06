import torch
import torch.nn as nn


class OneHotEncoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])

        t = t.long().contiguous().view(-1)
        ones = torch.sparse.torch.eye(self.n_classes).type_as(t)
        out = ones.index_select(0, t).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).contiguous().float()

        return out
