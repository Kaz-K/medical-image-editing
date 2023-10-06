import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLoss(nn.Module):

    epsilon = 1e-6

    def __init__(self,
                 dict_size: int,
                 margin: float,
                 use_distance_loss: bool,
                 use_regularization_loss: bool,
                 ):
        super().__init__()

        self.margin = margin
        self.use_distance_loss = use_distance_loss
        self.use_regularization_loss = use_regularization_loss

    def forward(self, embed_1, r_ids_1, embed_2, r_ids_2, codebook):
        # (b, n_features, n_loc)
        embed_1 = self._reshape_embed(embed_1)
        embed_2 = self._reshape_embed(embed_2)

        # (b, n_clusters, n_loc)
        r_ids_1 = self._reshape_ids(r_ids_1)
        r_ids_2 = self._reshape_ids(r_ids_2)

        l_cross_1 = self._calc_cross_loss(embed_1, r_ids_2, codebook)
        l_cross_2 = self._calc_cross_loss(embed_2, r_ids_1, codebook)

        l_cross = l_cross_1 + l_cross_2

        l_dist = 0.0
        if self.use_distance_loss:
            l_dist = self._calc_distance_loss(codebook)

        l_reg = 0.0
        if self.use_regularization_loss:
            l_reg = self._calc_regularization_loss(codebook)

        return l_cross, l_dist, l_reg

    def _calc_cross_loss(self, embed, r_ids, codebook):
        b, n_features, n_loc = embed.size()
        n_clusters = r_ids.size(1)

        # (n_features, n_clusters) -> (1, n_features, n_clusters, 1)
        centroid = codebook.detach().unsqueeze(0).unsqueeze(3)
        # (b, n_features, n_clusters, n_loc)
        centroid = centroid.expand(b, n_features, n_clusters, n_loc)
        # (b, n_features, n_clusters, n_loc)
        embed = embed.unsqueeze(2).expand(b, n_features, n_clusters, n_loc)
        # (b, n_clusters, n_loc)
        cross_dist = (torch.norm((embed - centroid), 2, 1) ** 2) * r_ids

        # (b, n_clusters)
        absence_ids = r_ids.sum(2) == 0

        # (b, n_clusters)
        cross_dist = cross_dist.sum(2) / (r_ids.sum(2) + self.epsilon)
        cross_dist = cross_dist[absence_ids == False].mean()

        return cross_dist

    def _calc_distance_loss(self, codebook):
        n_features, n_clusters = codebook.size()

        # (n_features, n_clusters, n_clusters)
        centroid_a = codebook.unsqueeze(2).expand(n_features, n_clusters, n_clusters)
        centroid_b = centroid_a.permute(0, 2, 1)
        diff = centroid_a - centroid_b

        c_dist = torch.sum(
            torch.clamp(2 * self.margin - torch.norm(diff, 2, 0), min=0) ** 2,
            dim=[0, 1],
        )

        dist_loss = c_dist / (2 * n_clusters * (n_clusters - 1))
        dist_loss = dist_loss.mean()

        return dist_loss

    def _calc_regularization_loss(self, codebook):
        reg_loss = torch.mean(torch.norm(codebook, 2, 0))
        return reg_loss

    def _reshape_embed(self, embed):
        b, n_features, h, w = embed.size()
        n_loc = h * w

        # (b, n_features, n_loc)
        embed = embed.contiguous().view(b, n_features, n_loc)
        return embed

    def _reshape_ids(self, ids):
        # (b, n_clusters, h, w)
        # ids = self.one_hot_encoder(ids)
        b, n_clusters, h, w = ids.size()
        n_loc = h * w

        # (b, n_clusters, n_loc)
        ids = ids.contiguous().view(b, n_clusters, n_loc)
        return ids
