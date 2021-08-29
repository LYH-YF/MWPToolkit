# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 09:10:50
# @File: seq_attention.py


import math

import torch
from torch import nn
from torch.nn import functional as F

class SeqAttention(nn.Module):
    def __init__(self, hidden_size,context_size):
        super(SeqAttention, self).__init__()
        self.hidden_size=hidden_size
        self.context_size=context_size

        self.linear_out = nn.Linear(hidden_size*2, context_size)

    def forward(self, inputs, encoder_outputs,mask):
        """
        Args:
            inputs (torch.Tensor): shape [batch_size, 1, hidden_size].
            encoder_outputs (torch.Tensor): shape [batch_size, sequence_length, hidden_size].

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output, shape [batch_size, 1, context_size].
                attention, shape [batch_size, 1, sequence_length].
        """
        batch_size = inputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        attn = torch.bmm(inputs, encoder_outputs.transpose(1,2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, seq_length), dim=1).view(batch_size, -1, seq_length)

        mix = torch.bmm(attn, encoder_outputs)

        combined = torch.cat((mix, inputs), dim=2)

        output = torch.tanh(self.linear_out(combined.view(-1, 2*self.hidden_size)))\
                            .view(batch_size, -1, self.context_size)

        return output, attn

class Attention(nn.Module):
    """ Calculate attention

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """
    def __init__(self, dim_value, dim_query,
                 dim_hidden=256, dropout_rate=0.5):
        super(Attention, self).__init__()
        self.relevant_score = \
            MaskedRelevantScore(dim_value, dim_query, dim_hidden)

    def forward(self, value, query, lens):
        """ Generate variable embedding with attention.

        Args:
            query (FloatTensor): Current hidden state, with size [batch_size, dim_query].
            value (FloatTensor): Sequence to be attented, with size [batch_size, seq_len, dim_value].
            lens (list of int): Lengths of values in a batch.

        Return:
            FloatTensor: Calculated attention, with size [batch_size, dim_value].
        """
        relevant_scores = self.relevant_score(value, query, lens)
        e_relevant_scores = torch.exp(relevant_scores)
        weights = e_relevant_scores / e_relevant_scores.sum(-1, keepdim=True)
        attention = (weights.unsqueeze(-1) * value).sum(1)
        return attention


class MaskedRelevantScore(nn.Module):
    """ Relevant score masked by sequence lengths.

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """
    def __init__(self, dim_value, dim_query, dim_hidden=256,
                 dropout_rate=0.0):
        super(MaskedRelevantScore, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.relevant_score = RelevantScore(dim_value, dim_query,
                                            dim_hidden,
                                            dropout_rate)

    def forward(self, value, query, lens):
        """ Choose candidate from candidates.

        Args:
            query (torch.FloatTensor): Current hidden state, with size [batch_size, dim_query].
            value (torch.FloatTensor): Sequence to be attented, with size [batch_size, seq_len, dim_value].
            lens (list of int): Lengths of values in a batch.

        Return:
            torch.Tensor: Activation for each operand, with size [batch, max([len(os) for os in operands])].
        """
        relevant_scores = self.relevant_score(value, query)

        # make mask to mask out padding embeddings
        mask = torch.zeros_like(relevant_scores)
        for b, n_c in enumerate(lens):
            mask[b, n_c:] = -math.inf

        # apply mask
        relevant_scores += mask

        return relevant_scores


class RelevantScore(nn.Module):
    def __init__(self, dim_value, dim_query, hidden1, dropout_rate=0):
        super(RelevantScore, self).__init__()
        self.lW1 = nn.Linear(dim_value, hidden1, bias=False)
        self.lW2 = nn.Linear(dim_query, hidden1, bias=False)
        self.b = nn.Parameter(
            torch.normal(torch.zeros(1, 1, hidden1), 0.01))
        self.tanh = nn.Tanh()
        self.lw = nn.Linear(hidden1, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, query):
        """
        Args:
            value (torch.FloatTensor): shape [batch, seq_len, dim_value].
            query (torch.FloatTensor): shape [batch, dim_query].
        """
        u = self.tanh(self.dropout(
            self.lW1(value)
            + self.lW2(query).unsqueeze(1)
            + self.b))
        # u.size() == (batch, seq_len, dim_hidden)
        return self.lw(u).squeeze(-1)
