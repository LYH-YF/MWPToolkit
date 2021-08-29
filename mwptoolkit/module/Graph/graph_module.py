# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:00:28
# @File: graph_module.py


import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Layer.graph_layers import PositionwiseFeedForward,LayerNorm
from mwptoolkit.module.Graph.gcn import GCN

class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        """
        Args:
            indim: dimensionality of input node features
            hiddim: dimensionality of the joint hidden embedding
            outdim: dimensionality of the output node features
            combined_feature_dim: dimensionality of the joint hidden embedding for graph
            K: number of graph nodes/objects on the image
        """
        self.in_dim = indim
        self.h = 4
        self.d_k = outdim//self.h
        
        self.graph = nn.ModuleList()
        for _ in range(self.h):
            self.graph.append(GCN(indim,hiddim,self.d_k,dropout))
        
        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        """
        Args:
            graph_nodes (torch.Tensor): input features, shape [batch_size, node_num, in_feat_dim]
        
        Returns:
            torch.Tensor: adjacency matrix, shape [batch_size, node_num, node_num]
        """
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        
        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        
        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        
        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix
    
    def normalize(self, A, symmetric=True):
        """
        Args:
            A (torch.Tensor): adjacency matrix (node_num, node_num)
        
        Returns:
            adjacency matrix (node_num, node_num) 
        """
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(A)
       
    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        """
        Args:
            graph_nodes (torch.Tensor):input features, shape [batch_size, node_num, in_feat_dim]
        
        Returns:
            torch.Tensor: graph_encode_features, shape [batch_size, node_num, out_feat_dim]
        """
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            adj_list = [adj,adj,adj,adj]
        else:
            adj = graph.float()
            adj_list = [adj[:,1,:],adj[:,1,:],adj[:,4,:],adj[:,4,:]]

        g_feature = \
            tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
        
        g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
        
        graph_encode_features = self.feed_foward(g_feature) + g_feature
        
        return adj, graph_encode_features

class Parse_Graph_Module(nn.Module):
    def __init__(self, hidden_size):
        super(Parse_Graph_Module, self).__init__()
        
        self.hidden_size = hidden_size
        self.node_fc1 = nn.Linear(hidden_size, hidden_size)
        self.node_fc2 = nn.Linear(hidden_size, hidden_size)
        self.node_out = nn.Linear(hidden_size * 2, hidden_size)
    
    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(graph)
        
    def forward(self, node, graph):
        graph = graph.float()
        batch_size = node.size(0)
        for i in range(batch_size):
            graph[i] = self.normalize(graph[i])
        
        node_info = torch.relu(self.node_fc1(torch.matmul(graph, node)))
        node_info = torch.relu(self.node_fc2(torch.matmul(graph, node_info)))
        
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        
        return agg_node_info


class Num_Graph_Module(nn.Module):
    def __init__(self, node_dim):
        super(Num_Graph_Module, self).__init__()
        
        self.node_dim = node_dim
        self.node1_fc1 = nn.Linear(node_dim, node_dim)
        self.node1_fc2 = nn.Linear(node_dim, node_dim)
        self.node2_fc1 = nn.Linear(node_dim, node_dim)
        self.node2_fc2 = nn.Linear(node_dim, node_dim)
        self.graph_weight = nn.Linear(node_dim * 4, node_dim)
        self.node_out = nn.Linear(node_dim * 2, node_dim)
    
    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(graph)

    def forward(self, node, graph1, graph2):
        graph1 = graph1.float()
        graph2 = graph2.float()
        batch_size = node.size(0)
        
        for i in range(batch_size):
            graph1[i] = self.normalize(graph1[i], False)
            graph2[i] = self.normalize(graph2[i], False)
        
        node_info1 = torch.relu(self.node1_fc1(torch.matmul(graph1, node)))
        node_info1 = torch.relu(self.node1_fc2(torch.matmul(graph1, node_info1)))
        node_info2 = torch.relu(self.node2_fc1(torch.matmul(graph2, node)))
        node_info2 = torch.relu(self.node2_fc2(torch.matmul(graph2, node_info2)))
        gate = torch.cat((node_info1, node_info2, node_info1+node_info2, node_info1-node_info2), dim=2)
        gate = torch.sigmoid(self.graph_weight(gate))
        node_info = gate * node_info1 + (1-gate) * node_info2
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        
        return agg_node_info
