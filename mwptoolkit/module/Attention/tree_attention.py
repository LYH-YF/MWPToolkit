# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 09:11:19
# @File: tree_attention.py


import torch
from torch import nn


class TreeAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        """
        Args:
            hidden (torch.Tensor): hidden representation, shape [1, batch_size, hidden_size]
            encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size]. 
            seq_mask (torch.Tensor): sequence mask, shape [batch_size, sequence_length].
        
        Returns:
            attn_energies (torch.Tensor): attention energies, shape [batch_size, 1, sequence_length].
        """
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)