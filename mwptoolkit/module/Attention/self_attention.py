# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:44:59
# @File: self_attention.py


import torch
from torch import nn

class SelfAttentionMask(nn.Module):
    def __init__(self, init_size=100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)

    @staticmethod
    def get_mask(size):
        weights = torch.ones((size, size), dtype=torch.uint8).triu_(1)  # above the diagonal == 1
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        masks = self.weights[:size, :size].detach()
        return masks
