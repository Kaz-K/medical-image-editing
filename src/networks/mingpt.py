# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import top_k_top_p_filtering


logger = logging.getLogger(__name__)


class GPTConfig:
    emb_pdrop = 0.1
    res_pdrop = 0.1
    att_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embed = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # key, query, value projections for all heads
        self.k = nn.Linear(config.n_embed, config.n_embed)
        self.q = nn.Linear(config.n_embed, config.n_embed)
        self.v = nn.Linear(config.n_embed, config.n_embed)

        # regularization
        self.att_drop = nn.Dropout(config.att_pdrop)
        self.res_drop = nn.Dropout(config.res_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embed, config.n_embed)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))

        if hasattr(config, 'n_unmasked'):
            mask[:config.n_unmasked, :config.n_unmasked] = 1

        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, value for all heads in batch and move forward to be the batch dim
        k = self.k(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.v(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))  # (2, B, nh, T, hs)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)  # (B, nh, 2 * T, hs)
            v = torch.cat((past_v, v), dim=-2)  # (B, nh, 2 * T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all heads outputs side by side

        # output projection
        y = self.res_drop(self.proj(y))

        return y, present


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.att = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.res_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present:
            assert not self.training

        att, present = self.att(self.ln1(x), layer_past=layer_past)

        x = x + att
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present

        return x


class GPT(nn.Module):
    """ the full GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size,
                 block_size,
                 n_layer=12,
                 n_head=8,
                 n_embed=256,
                 emb_pdrop=0.0,
                 res_pdrop=0.0,
                 att_pdrop=0.0,
                 n_unmasked=0,
                 ):
        super().__init__()
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_pdrop=emb_pdrop,
            res_pdrop=res_pdrop,
            att_pdrop=att_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embed=n_embed,
            n_unmasked=n_unmasked,
        )

        # input embedding stem
        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        self.drop = nn.Dropout(config.emb_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None):
        # forward the GPT model
        token_embeddings = self.tok_embed(idx)  # each index maps to a (learnable) vector

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, 'Cannot forward, model block size is exhausted.'

        position_embeddings = self.pos_embed[:, :t, :]  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def forward_with_past(self, idx, embeddings=None, past=None, past_length=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_embed(idx)  # each index maps to a (learnable) vector
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)  # (n_layer, 2, b, nh, len_past, dim_head)
            past_shape = list(past.shape)
            expected_shape = [
                self.config.n_layer, 2, idx.shape[0], self.config.n_head,
                past_length, self.config.n_embed // self.config.n_head,
            ]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_embed[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_embed[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            present.append(present)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, torch.stack(presents)
