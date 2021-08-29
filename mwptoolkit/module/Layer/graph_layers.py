# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:00:42
# @File: graph_layers.py


import math
import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input variable.
        
        Returns:
            torch.Tensor: output variable.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input variable.
        
        Returns:
            torch.Tensor: output variable.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, concat = False):
        super(MeanAggregator, self).__init__()
        
        self.concat = concat
        self.fc_x = nn.Linear(input_dim, output_dim, bias=True)
        self.activation = activation

    def forward(self, inputs):
        x, neibs, _ = inputs
        agg_neib = neibs.mean(dim=1)
        if self.concat:
            out_tmp = torch.cat([x, agg_neib],dim=1)
            out = self.fc_x(out_tmp)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)
        if self.activation:
            out = self.activation(out)
        return out
    


# Graph_Conv
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
