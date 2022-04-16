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


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        #all_problem_hidden S*B*H  problem_num_mask B*S
        batch_size=inputs.size(1)
        max_len = inputs.size(0)

        repeat_dims1=[1,1,max_len,1]
        repeat_dims2=[1,max_len,1,1]
        sen1=inputs.transpose(0,1).unsqueeze(2)#B*S*1*H
        sen2=inputs.transpose(0,1).unsqueeze(1)#B*1*S*H

        sen1=sen1.repeat(repeat_dims1)
        sen2=sen2.repeat(repeat_dims2)#S*S*B*H

        energy_in=torch.cat((sen1, sen2), 3)#B*S*S*2H
        score_feature = torch.tanh(self.attn(energy_in))#B*S*S*H
        attn_energies = self.score(score_feature).squeeze(3)  # B*S*S

        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S*S
        return attn_energies
