# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:49:49
# @File: gcn.py


import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Layer.graph_layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): input features, shape [batch_size, node_num, in_feat_dim]
            adj (torch.Tensor): adjacency matrix, shape [batch_size, node_num, node_num]
        
        Returns:
            torch.Tensor: gcn_enhance_feature, shape [batch_size, node_num, out_feat_dim]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    