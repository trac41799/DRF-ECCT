from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from utils import *

torch.cuda.is_available()


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers =  clones(layer, N)
        self.norm = LayerNorm(layer.size)
        #if N > 1:
        #    self.norm2 = LayerNorm(layer.size)
        self.feed_forward = clones(PositionwiseFeedForward(layer.size,
                                                          layer.size*4), 2)

    def forward(self, x, mask, flag=False):
        for idx, layer in enumerate(self.layers, start=1):
          #if idx in [1,6]:
          #  x = layer(x, mask, self.feed_forward[0], flag)
          x = layer(x, mask, self.feed_forward[int(idx%2)], flag)  #self.feed_forward[wdx]
          #x = layer(x, mask, None, flag)
          #if idx == len(self.layers) //2 and len(self.layers) > 1:
          #  x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, h, size, dropout):
        super(EncoderLayer, self).__init__()
        self.h = h
        self.self_attn = MultiHeadedAttention(h, size)
        #self.feed_forward = PositionwiseFeedForward(size,
        #                                            size*4, dropout)
        #self.norm = LayerNorm(size)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, ffn_blk = None, flag=False):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, flag))
        return self.sublayer[1](x, ffn_blk)


class DRF(nn.Module):
    def __init__(self, h, d_k):
        super(DRF, self).__init__()
        #self.mish = torch.nn.Mish()
        #self.w = torch.nn.Parameter(torch.empty(96,96))
        self.w = torch.nn.Parameter(torch.empty(h,d_k)) #,d_k//2)) #


    def forward(self, x, mask=None):
        with torch.no_grad():
          tau = torch.abs(x.mean(dim=-1, keepdim=True))
        x = torch.matmul(x - tau, self.w.unsqueeze(0).unsqueeze(-1))
        return F.gelu(x) #.masked_fill(mask[:,:,:,:x.size(-1)], -1e9))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.drf = DRF(h, self.d_k)
        #self.drf2 = DRF(h, 1, False)

    def forward(self, query, key, value, mask=None, flag=False):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears,
                             (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask)
        if flag:
          self.attn = attn

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(self.drf(query, mask), self.drf(key, mask).transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores / math.sqrt(d_k), dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 =  nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        code = args.code

        self.src_embed = torch.nn.Parameter(torch.eye(
            code.n + code.pc_matrix.size(0), args.d_model))
        self.decoder = Encoder(EncoderLayer(args.h,
                                args.d_model, dropout,), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.src_mask = self.get_mask(code)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome, flag=False):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask, flag)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z, x, y):
        z_mul = y*x
        cce_loss = F.binary_cross_entropy_with_logits(
            z, sign_to_bin(torch.sign(z_mul)))
        loss = cce_loss
        x_pred = sign_to_bin(torch.sign(-z * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        mask_size = code.n + code.pc_matrix.size(0)
        mask = torch.eye(mask_size, mask_size)
        for ii in range(code.pc_matrix.size(0)):
            idx = torch.where(code.pc_matrix[ii] > 0)[0]
              for jj in idx:
                  for kk in idx:
                      if jj != kk:
                          mask[jj, kk] += 1
                          mask[kk, jj] += 1
                          mask[code.n + ii, jj] += 1
                          mask[jj, code.n + ii] += 1
        src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
        return src_mask


############################################################
############################################################

if __name__ == '__main__':
    pass