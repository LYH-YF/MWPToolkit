# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:48:33
# @File: graph_based_encoder.py


import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Graph.graph_module import Graph_Module,Parse_Graph_Module,Num_Graph_Module
from mwptoolkit.module.Layer.graph_layers import MeanAggregator
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.utils.utils import clones

def replace_masked_values(tensor, mask, replace_with):
    return tensor.masked_fill((1 - mask).bool(), replace_with)
class GraphBasedEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size,rnn_cell_type,bidirectional, num_layers=2, dropout_ratio=0.5):
        super(GraphBasedEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_embedding, input_lengths, batch_graph, hidden=None):
        """
        Args:
            input_embedding (torch.Tensor): input variable, shape [sequence_length, batch_size, embedding_size].
            input_lengths (torch.Tensor): length of input sequence, shape: [batch_size].
            batch_graph (torch.Tensor): graph input variable, shape [batch_size, 5, sequence_length, sequence_length].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                pade_outputs, encoded variable, shape [sequence_length, batch_size, hidden_size].
                problem_output, vector representation of problem, shape [batch_size, hidden_size]. 
        """
        # Note: we run this all at once (over multiple batches of multiple sequences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_embedding, input_lengths, enforce_sorted=True)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.encoder(packed, pade_hidden)
        pade_outputs, hidden_states = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output

class GraphEncoder(nn.Module):
    def __init__(self, vocab_size,embedding_size,hidden_size,sample_size,sample_layer,bidirectional,dropout_ratio):
        super(GraphEncoder, self).__init__()
        self.hidden_size=hidden_size
        self.sample_size=sample_size
        self.sample_layer=sample_layer
        self.bidirectional=bidirectional
        self.embedding_size=embedding_size

        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding = BasicEmbedder(vocab_size,self.embedding_size,dropout_ratio,padding_idx=0)

        self.fw_aggregators = nn.ModuleList()
        self.bw_aggregators = nn.ModuleList()
        for layer in range(7):
            self.fw_aggregators.append(
                MeanAggregator(2*self.hidden_size, self.hidden_size, concat=True)
            )
            self.bw_aggregators.append(
                MeanAggregator(2*self.hidden_size, self.hidden_size, concat=True)
            )

        # self.fw_aggregator_0 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_1 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_2 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_3 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_4 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_5 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregator_6 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)

        # self.bw_aggregator_0 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_1 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_2 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_3 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_4 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_5 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.bw_aggregator_6 = MeanAggregator(
        #     2*self.hidden_size, self.hidden_size, concat=True)
        # self.fw_aggregators = [self.fw_aggregator_0, self.fw_aggregator_1, self.fw_aggregator_2,
        #                        self.fw_aggregator_3, self.fw_aggregator_4, self.fw_aggregator_5, self.fw_aggregator_6]
        # self.bw_aggregators = [self.bw_aggregator_0, self.bw_aggregator_1, self.bw_aggregator_2,
        #                        self.bw_aggregator_3, self.bw_aggregator_4, self.bw_aggregator_5, self.bw_aggregator_6]

        self.Linear_hidden = nn.Linear(
            2 * self.hidden_size, self.hidden_size)

        #self.concat = opt.concat

        # self.using_gpu = False
        # if self.opt.gpuid > -1:
        #     self.using_gpu = True

        self.embedding_bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size//2, bidirectional=True, bias = True, batch_first = True, dropout= dropout_ratio, num_layers=1)
        self.padding_vector = torch.randn(1,self.hidden_size, dtype = torch.float, requires_grad=True)

    def forward(self, fw_adj_info, bw_adj_info, feature_info, batch_nodes):
        #fw_adj_info, bw_adj_info, feature_info, batch_nodes = graph_batch

        # print self.hidden_layer_dim

        # if self.using_gpu:
        #     fw_adj_info = fw_adj_info.cuda()
        #     bw_adj_info = bw_adj_info.cuda()
        #     feature_info = feature_info.cuda()
        #     batch_nodes = batch_nodes.cuda()
        device = batch_nodes.device

        #feature_by_sentence = feature_info[:-1,:].view(batch_nodes.size()[0], -1)
        #feature_sentence_vector = self.embedding(feature_by_sentence)
        feature_sentence_vector = self.embedding(feature_info)
        #feature_sentence_vector = self.input_dropout(feature_sentence_vector)
        
        output_vector, (ht,_) = self.embedding_bilstm(feature_sentence_vector)

        feature_vector = output_vector.contiguous().view(-1, self.hidden_size)
        # if self.using_gpu:
        #     feature_embedded = torch.cat([feature_vector, self.padding_vector.cuda()], 0)
        # else:
        #     feature_embedded = torch.cat([feature_vector, self.padding_vector], 0)
        #feature_embedded = torch.cat([feature_vector, self.padding_vector.to(device)], 0)
        feature_embedded=feature_vector

        batch_size = feature_embedded.size()[0]
        node_repres = feature_embedded.view(batch_size, -1)

        #fw_sampler = UniformNeighborSampler(fw_adj_info)
        #bw_sampler = UniformNeighborSampler(bw_adj_info)
        nodes = batch_nodes.long().view(-1, )

        fw_hidden = F.embedding(nodes, node_repres)
        bw_hidden = F.embedding(nodes, node_repres)

        #fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        #bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = torch.tensor(0)
        bw_sampled_neighbors_len = torch.tensor(0)

        # sampler
        fw_tmp = fw_adj_info[nodes]
        fw_perm = torch.randperm(fw_tmp.size(1))
        fw_tmp = fw_tmp[:,fw_perm]
        fw_sampled_neighbors = fw_tmp[:,:self.sample_size]

        bw_tmp = bw_adj_info[nodes]
        bw_perm = torch.randperm(bw_tmp.size(1))
        bw_tmp = bw_tmp[:,bw_perm]
        bw_sampled_neighbors = bw_tmp[:,:self.sample_size]

        # begin sampling
        for layer in range(self.sample_layer):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 1
            # if self.using_gpu and layer <= 6:
            #     self.fw_aggregators[layer] = self.fw_aggregators[layer].cuda()
            if layer == 0:
                neigh_vec_hidden = F.embedding(
                    fw_sampled_neighbors, node_repres)
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
            else:
                # if self.using_gpu:
                #     neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                #         [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                # else:
                #     neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                #         [1, dim_mul * self.hidden_layer_dim])], 0))
                neigh_vec_hidden = F.embedding(
                    fw_sampled_neighbors,
                    torch.cat(
                        [fw_hidden, torch.zeros([1, dim_mul * self.hidden_size]).to(device)],dim=0
                    )
                )

            if layer > 6:
                    fw_hidden = self.fw_aggregators[6](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            else:
                    fw_hidden = self.fw_aggregators[layer](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.bidirectional:
                # if self.using_gpu and layer <= 6:
                #     self.bw_aggregators[layer] = self.bw_aggregators[layer].cuda(
                #     )

                if layer == 0:
                    neigh_vec_hidden = F.embedding(
                        bw_sampled_neighbors, node_repres)
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                    tmp_mask = torch.sign(tmp_sum)
                    bw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
                else:
                    # if self.using_gpu:
                    #     neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                    #         [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                    # else:
                    #     neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                    #         [1, dim_mul * self.hidden_layer_dim])], 0))
                    neigh_vec_hidden = F.embedding(
                        bw_sampled_neighbors,
                        torch.cat(
                            [bw_hidden, torch.zeros([1, dim_mul * self.hidden_size]).to(device)], dim=0
                        )
                    )
                bw_hidden = self.dropout(bw_hidden)
                neigh_vec_hidden = self.dropout(neigh_vec_hidden)

                if layer > 6:
                    bw_hidden = self.bw_aggregators[6](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                else:
                    bw_hidden = self.bw_aggregators[layer](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        fw_hidden = fw_hidden.view(-1, batch_nodes.size()
                                   [1], self.hidden_size)

        if self.bidirectional:
            bw_hidden = bw_hidden.view(-1, batch_nodes.size()
                                       [1], self.hidden_size)
            hidden = torch.cat([fw_hidden, bw_hidden], 2)
        else:
            hidden = fw_hidden

        pooled = torch.max(hidden, 1)[0]
        graph_embedding = pooled.view(-1, self.hidden_size)

        return hidden, graph_embedding, output_vector


class GraphBasedMultiEncoder(nn.Module):
    def __init__(self, input1_size, input2_size, embed_model, embedding1_size, 
                 embedding2_size, hidden_size, n_layers=2, hop_size=2, dropout=0.5):
        super(GraphBasedMultiEncoder, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.embedding1_size = embedding1_size
        self.embedding2_size = embedding2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.hop_size = hop_size

        self.embedding1 = embed_model
        self.embedding2 = nn.Embedding(input2_size, embedding2_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding1_size+embedding2_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.parse_gnn = clones(Parse_Graph_Module(hidden_size), hop_size)
        
    def forward(self, input1_var, input2_var, input_length, parse_graph, hidden=None):
        """
        """
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded1 = self.embedding1(input1_var)  # S x B x E
        embedded2 = self.embedding2(input2_var)
        embedded = torch.cat((embedded1, embedded2), dim=2)
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        pade_outputs = pade_outputs.transpose(0, 1)
        for i in range(self.hop_size):
            pade_outputs = self.parse_gnn[i](pade_outputs, parse_graph[:,2])
        pade_outputs = pade_outputs.transpose(0, 1)
        #problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        
        return pade_outputs, pade_hidden


class NumEncoder(nn.Module):
    def __init__(self, node_dim, hop_size=2):
        super(NumEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.hop_size = hop_size
        self.num_gnn = clones(Num_Graph_Module(node_dim), hop_size)
    
    def forward(self, encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad):
        num_embedding = num_encoder_outputs.clone()
        batch_size = num_embedding.size(0)
        num_mask = (num_pos_pad > -1).long()
        node_mask = (num_order_pad > 0).long()
        greater_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) > \
                        num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        lower_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) <= \
                        num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        greater_graph_mask = greater_graph_mask.long()
        lower_graph_mask = lower_graph_mask.long()
        
        diagmat = torch.diagflat(torch.ones(num_embedding.size(1), dtype=torch.long, device=num_embedding.device))
        diagmat = diagmat.unsqueeze(0).expand(num_embedding.size(0), -1, -1)
        graph_ = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1-diagmat)
        graph_greater = graph_ * greater_graph_mask + diagmat
        graph_lower = graph_ * lower_graph_mask + diagmat
        
        for i in range(self.hop_size):
            num_embedding = self.num_gnn[i](num_embedding, graph_greater, graph_lower)
        
        #        gnn_info_vec = torch.zeros((batch_size, 1, encoder_outputs.size(-1)),
        #                                   dtype=torch.float, device=num_embedding.device)
        #        gnn_info_vec = torch.cat((encoder_outputs.transpose(0, 1), gnn_info_vec), dim=1)
        gnn_info_vec = torch.zeros((batch_size, encoder_outputs.size(0)+1, encoder_outputs.size(-1)),
                                   dtype=torch.float, device=num_embedding.device)
        clamped_number_indices = replace_masked_values(num_pos_pad, num_mask, gnn_info_vec.size(1)-1)
        gnn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, num_embedding.size(-1)), num_embedding)
        gnn_info_vec = gnn_info_vec[:, :-1, :]
        gnn_info_vec = gnn_info_vec.transpose(0, 1)
        gnn_info_vec = encoder_outputs + gnn_info_vec
        num_embedding = num_encoder_outputs + num_embedding
        problem_output = torch.max(gnn_info_vec, 0).values
        
        return gnn_info_vec, num_embedding, problem_output


# class EEHEncoder(nn.Module):
#     def __init__(self,
#                  input_size,
#                  embedding_size,
#                  pos_embedding_size,
#                  hidden_size,
#                  category_size,
#                  pretrain_emb,
#                  edge_size,
#                  n_layers=2,
#                  dropout=0.5,
#                  use_self_attention=False,
#                  use_glove_embedding=False,
#                  use_kas2t_encoder=False,
#                  use_g2t_stanford=False,
#                  use_gate=False,
#                  use_cate=False,
#                  use_compare=False,
#                  use_number_encoder=False
#                  ):
#         super(EEHEncoder, self).__init__()
#
#         self.input_size = input_size
#         self.embedding_size = embedding_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = dropout
#         self.shrink_size = hidden_size
#         self.pos_embedding_size = pos_embedding_size
#         BasicRNNEncoder = importlib.import_module('mwptoolkit.module.Encoder.rnn_encoder','BasicRNNEncoder')
#         if use_number_encoder == True:
#             self.shrink_size = self.shrink_size - pos_embedding_size
#         if use_self_attention == True:
#             self.shrink_size = self.shrink_size - pos_embedding_size
#
#         self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
#         if use_glove_embedding == True:
#             self.embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
#             self.embedding.weight.requires_grad = False
#
#         self.em_dropout = nn.Dropout(dropout)
#         # self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
#
#         self.edge_embedding = nn.Embedding(edge_size, 32, padding_idx=0)
#         self.edge_dense1 = nn.Linear(384, 32)
#         self.edge_dense2 = nn.Linear(384, 32)
#         if use_kas2t_encoder == False:
#             self.rnn_encoder = BasicRNNEncoder()
#             if use_g2t_stanford == True:
#                 self.gcn = Graph_Module(self.shrink_size, self.shrink_size, self.shrink_size)
#         else:
#             self.category_embedding = nn.Embedding(category_size, 384, padding_idx=0)
#             self.dis_embedding = DisPositionalEncoding(384, 160)
#             self.dis_embedding1 = DisPositionalEncoding(384, 160)
#             # self.dis_embedding = Dis_PositionalEncoding(384, 160)
#
#             self.gru_pade = nn.GRU(embedding_size, 384, n_layers, dropout=dropout, bidirectional=True)
#
#             if use_g2t_stanford == True:
#                 self.gcn = Graph_Module(384, 384, 384)
#             self.gat_n = [GraphAttentionLayer(384, 384, 0.5, 0.1, concat=False) for _ in range(8)]
#             for i, attention in enumerate(self.gat_n):
#                 self.add_module('attention_{}'.format(i), attention)
#
#             self.pos_embedding = PositionalEncoding(128, 160)
#
#             self.encoder_layers = EncoderLayer(self.shrink_size, 16, self.shrink_size, dropout)
#             # self.encoder_layers2 = EncoderLayer(self.shrink_size, 4, self.shrink_size, dropout)
#
#         if use_gate == True:
#             self.enc_linear = nn.Linear(self.shrink_size, self.shrink_size)
#             self.enc_dense = nn.Linear(self.shrink_size, 1)
#             self.num_pos_embedding = nn.Embedding(2, pos_embedding_size, padding_idx=0)
#
#         if use_cate == True:
#             self.type_linear = nn.Linear(self.shrink_size, self.shrink_size)
#             self.type_dense = nn.Linear(self.shrink_size, 5)
#             self.type_pos_embedding = nn.Embedding(5, pos_embedding_size, padding_idx=0)
#
#         if use_compare == True:
#             self.pair_score = nn.Linear(self.hidden_size, 1)
#
#         if use_number_encoder == True:
#             self.dense_emb = nn.Linear(self.embedding_size, self.pos_embedding_size)
#
#             self.nums_char_embedding = nn.Embedding(16, pos_embedding_size, padding_idx=0)
#             self.pade_embedding = nn.Embedding(2, pos_embedding_size, padding_idx=0)
#             self.num_gru = nn.GRU(pos_embedding_size, pos_embedding_size, n_layers, dropout=dropout, bidirectional=True)
#
#             self.self_attn = SelfAttention(pos_embedding_size)
