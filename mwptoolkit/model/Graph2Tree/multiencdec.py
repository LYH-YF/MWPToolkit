import copy
import random

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Encoder.graph_based_encoder import GraphBasedMultiEncoder, NumEncoder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
#from mwptoolkit.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from mwptoolkit.module.Layer.layers import AttnDecoderRNN
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Strategy.beam_search import TreeBeam, Beam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask
from mwptoolkit.utils.utils import copy_list,clones
USE_CUDA=torch.cuda.is_available()

class MultiEncDec(nn.Module):
    def __init__(self, config, dataset):
        super(MultiEncDec,self).__init__()
        self.device = config['device']
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.hop_size = config['hop_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.dropout_ratio = config['dropout_ratio']

        self.operator_nums = dataset.operator_nums
        self.generate_nums = len(dataset.generate_list)
        self.num_start1 = dataset.num_start1
        self.num_start2 = dataset.num_start2
        self.input1_size = len(dataset.in_idx2word_1)
        self.input2_size = len(dataset.in_idx2word_2)
        self.output2_size = len(dataset.out_idx2symbol_2)
        self.unk1 = dataset.out_symbol2idx_1[SpecialTokens.UNK_TOKEN]
        self.unk2 = dataset.out_symbol2idx_2[SpecialTokens.UNK_TOKEN]
        self.sos2 = dataset.out_symbol2idx_2[SpecialTokens.SOS_TOKEN]
        self.eos2 = dataset.out_symbol2idx_2[SpecialTokens.EOS_TOKEN]

        self.out_symbol2idx1 = dataset.out_symbol2idx_1
        self.out_idx2symbol1 = dataset.out_idx2symbol_1
        self.out_symbol2idx2 = dataset.out_symbol2idx_2
        self.out_idx2symbol2 = dataset.out_idx2symbol_2
        generate_list = dataset.generate_list
        self.generate_list = [self.out_symbol2idx1[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number

        try:
            self.out_sos_token1 = self.out_symbol2idx1[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token1 = None
        try:
            self.out_eos_token1 = self.out_symbol2idx1[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token1 = None
        try:
            self.out_pad_token1 = self.out_symbol2idx1[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token1 = None
        try:
            self.out_sos_token2 = self.out_symbol2idx2[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token2 = None
        try:
            self.out_eos_token2 = self.out_symbol2idx2[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token2 = None
        try:
            self.out_pad_token2 = self.out_symbol2idx2[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token2 = None
        # Initialize models
        embedder = BaiscEmbedder(self.input1_size, self.embedding_size, self.dropout_ratio)
        in_embedder = self._init_embedding_params(dataset.trainset,dataset.in_idx2word_1,config['embedding_size'],embedder)

        self.encoder = EncoderSeq(input1_size=self.input1_size, input2_size=self.input2_size, 
                         embed_model=in_embedder, embedding1_size=self.embedding_size, embedding2_size=self.embedding_size // 4, 
                         hidden_size=self.hidden_size, n_layers=self.n_layers, hop_size=self.hop_size)
        self.numencoder = NumEncoder(node_dim=self.hidden_size, hop_size=self.hop_size)
        self.predict = Prediction(hidden_size=self.hidden_size, op_nums=self.operator_nums,
                            input_size=self.generate_nums)
        self.generate = GenerateNode(hidden_size=self.hidden_size, op_nums=self.operator_nums, embedding_size=self.embedding_size)
        self.merge = Merge(hidden_size=self.hidden_size, embedding_size=self.embedding_size)
        self.decoder = AttnDecoderRNN_(self.hidden_size,
                                self.embedding_size,
                                self.output2_size,
                                self.output2_size,
                                self.n_layers,
                                self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()

    def _init_embedding_params(self,train_data,vocab,embedding_size,embedder):
        sentences=[]
        for data in train_data:
            sentence=[]
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentences.append(sentence)
        from gensim.models import word2vec
        model = word2vec.Word2Vec(sentences, size=embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        for idx in range(len(vocab)):
            if idx!=pad_idx:
                emb_vectors.append(np.array(model.wv[vocab[idx]]))
            else:
                emb_vectors.append(np.zeros((embedding_size)))
        emb_vectors=np.array(emb_vectors)
        embedder.embedder.weight.data.copy_(torch.from_numpy(emb_vectors))

        return embedder
    
    def calculate_loss(self,batch_data):
        input1_var = batch_data['input1']
        input2_var = batch_data['input2']
        input_length = batch_data['input1 len']
        target1 = batch_data['output1']
        target1_length = batch_data['output1 len']
        target2 = batch_data['output2']
        target2_length = batch_data['output2 len']
        num_stack_batch = batch_data['num stack']
        num_size_batch = batch_data['num size']
        generate_list = self.generate_list
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        parse_graph = batch_data['parse graph']
        equ_mask1 = batch_data['equ mask1']
        equ_mask2 = batch_data['equ mask2']

        unk1=self.unk1
        unk2=self.unk2
        num_start1=self.num_start1
        num_start2=self.num_start2
        sos2=self.sos2
        loss = train_double(
                input1_var, input2_var, input_length, target1, target1_length, target2, target2_length,
                num_stack_batch, num_size_batch, generate_list,
                self.encoder, self.numencoder, self.predict, self.generate, self.merge, self.decoder,unk1,unk2,num_start1,num_start2,sos2
                , num_pos_batch, num_order_batch, parse_graph, 
                beam_size=5, use_teacher_forcing=0.83, english=False)
        return loss

    def model_test(self,batch_data):
        input1_var = batch_data['input1']
        input2_var = batch_data['input2']
        input_length = batch_data['input1 len']
        target1 = batch_data['output1']
        target1_length = batch_data['output1 len']
        target2 = batch_data['output2']
        target2_length = batch_data['output2 len']
        num_stack_batch = batch_data['num stack']
        num_size_batch = batch_data['num size']
        generate_list = self.generate_list
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        parse_graph = batch_data['parse graph']
        equ_mask1 = batch_data['equ mask1']
        equ_mask2 = batch_data['equ mask2']
        num_list = batch_data['num list']

        unk1=self.unk1
        unk2=self.unk2
        num_start1=self.num_start1
        num_start2=self.num_start2
        sos2=self.sos2
        eos2=self.eos2

        result_type, test_res, score = evaluate_double(input1_var, input2_var, input_length, generate_list,num_start1,sos2,eos2,
                                                        self.encoder, self.numencoder, self.predict, self.generate, self.merge, self.decoder,
                                                        num_pos_batch, num_order_batch, parse_graph, beam_size=5)
        if result_type=="tree":
            output1=self.convert_idx2symbol1(test_res,num_list[0],copy_list(num_stack_batch[0]))
            targets1=self.convert_idx2symbol1(target1[0],num_list[0],copy_list(num_stack_batch[0]))
            return result_type, output1, targets1
        else:
            output2=self.convert_idx2symbol2(torch.tensor(test_res).view(1,-1),num_list,copy_list(num_stack_batch))
            targets2=self.convert_idx2symbol2(target2,num_list,copy_list(num_stack_batch))
            return result_type, output2, targets2
        
    def convert_idx2symbol1(self, output, num_list, num_stack):
        #batch_size=output.size(0)
        '''batch_size=1'''
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token1, self.out_eos_token1, self.out_pad_token1]:
                break
            symbol = self.out_idx2symbol1[idx]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list

    def convert_idx2symbol2(self, output, num_list, num_stack):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token2, self.out_eos_token2, self.out_pad_token2]:
                    break
                symbol = self.out_idx2symbol2[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                elif symbol == SpecialTokens.UNK_TOKEN:
                    try:
                        pos_list = num_stack[b_i].pop()
                        c = num_list[b_i][pos_list[0]]
                        res.append(c)
                    except:
                        res.append(symbol)
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

class EncoderSeq(nn.Module):
    def __init__(self, input1_size, input2_size, embed_model, embedding1_size, 
                 embedding2_size, hidden_size, n_layers=2, hop_size=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

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


class MultiEncDec_(nn.Module):
    def __init__(self, config, dataset):
        super(MultiEncDec,self).__init__()
        self.device = config['device']
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.hop_size = config['hop_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.dropout_ratio = config['dropout_ratio']

        self.operator_nums = dataset.operator_nums
        self.generate_nums = len(dataset.generate_list)
        self.num_start1 = dataset.num_start1
        self.num_start2 = dataset.num_start2
        self.input1_size = len(dataset.in_idx2word_1)
        self.input2_size = len(dataset.in_idx2word_2)
        self.output2_size = len(dataset.out_idx2symbol_2)
        self.unk1 = dataset.out_symbol2idx_1[SpecialTokens.UNK_TOKEN]
        self.unk2 = dataset.out_symbol2idx_2[SpecialTokens.UNK_TOKEN]
        self.sos2 = dataset.out_symbol2idx_2[SpecialTokens.SOS_TOKEN]
        self.eos2 = dataset.out_symbol2idx_2[SpecialTokens.EOS_TOKEN]

        self.out_symbol2idx1 = dataset.out_symbol2idx_1
        self.out_idx2symbol1 = dataset.out_idx2symbol_1
        self.out_symbol2idx2 = dataset.out_symbol2idx_2
        self.out_idx2symbol2 = dataset.out_idx2symbol_2
        generate_list = dataset.generate_list
        self.generate_list = [self.out_symbol2idx1[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number

        try:
            self.out_sos_token1 = self.out_symbol2idx1[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token1 = None
        try:
            self.out_eos_token1 = self.out_symbol2idx1[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token1 = None
        try:
            self.out_pad_token1 = self.out_symbol2idx1[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token1 = None
        try:
            self.out_sos_token2 = self.out_symbol2idx2[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token2 = None
        try:
            self.out_eos_token2 = self.out_symbol2idx2[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token2 = None
        try:
            self.out_pad_token2 = self.out_symbol2idx2[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token2 = None
        # Initialize models
        embedder = BaiscEmbedder(self.input1_size, self.embedding_size, self.dropout_ratio)
        in_embedder = self._init_embedding_params(dataset.trainset,dataset.in_idx2word_1,config['embedding_size'],embedder)

        #self.out_embedder = BaiscEmbedder(self.output2_size,self.embedding_size,self.dropout_ratio)

        self.encoder = GraphBasedMultiEncoder(input1_size=self.input1_size,
                                              input2_size=self.input2_size,
                                              embed_model=in_embedder,
                                              embedding1_size=self.embedding_size,
                                              embedding2_size=self.embedding_size // 4,
                                              hidden_size=self.hidden_size,
                                              n_layers=self.n_layers,
                                              hop_size=self.hop_size)

        self.numencoder = NumEncoder(node_dim=self.hidden_size, hop_size=self.hop_size)

        self.predict = TreeDecoder(hidden_size=self.hidden_size, op_nums=self.operator_nums, generate_size=self.generate_nums)

        self.generate = NodeGenerater(hidden_size=self.hidden_size, op_nums=self.operator_nums, embedding_size=self.embedding_size)

        self.merge = SubTreeMerger(hidden_size=self.hidden_size, embedding_size=self.embedding_size)

        self.decoder = AttnDecoderRNN(
            self.hidden_size,
            self.embedding_size,
            self.output2_size,
            self.output2_size,
            self.n_layers,
            self.dropout_ratio
        )

        # self.decoder = AttentionalRNNDecoder(embedding_size=self.embedding_size,
        #                                      hidden_size=self.hidden_size,
        #                                      context_size=self.hidden_size,
        #                                      num_dec_layers=self.n_layers,
        #                                      rnn_cell_type=self.rnn_cell_type,
        #                                      dropout_ratio=self.dropout_ratio)
        #self.out = nn.Linear(self.hidden_size, self.output2_size)

        self.loss = MaskedCrossEntropyLoss()

    def _init_embedding_params(self,train_data,vocab,embedding_size,embedder):
        sentences=[]
        for data in train_data:
            sentence=[]
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentences.append(sentence)
        from gensim.models import word2vec
        model = word2vec.Word2Vec(sentences, size=embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        for idx in range(len(vocab)):
            if idx!=pad_idx:
                emb_vectors.append(np.array(model.wv[vocab[idx]]))
            else:
                emb_vectors.append(np.zeros((embedding_size)))
        emb_vectors=np.array(emb_vectors)
        embedder.embedder.weight.data.copy_(torch.from_numpy(emb_vectors))

        return embedder
    

    def forward(self,input1_var, input2_var, input_length, target1, target1_length, target2, target2_length,\
                num_stack_batch, num_size_batch,generate_list,num_pos_batch, num_order_batch, parse_graph):
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_list)
        for i in num_size_batch:
            d = i + len(generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask)

        num_pos_pad = []
        max_num_pos_size = max(num_size_batch)
        for i in range(len(num_pos_batch)):
            temp = num_pos_batch[i] + [-1] * (max_num_pos_size - len(num_pos_batch[i]))
            num_pos_pad.append(temp)
        num_pos_pad = torch.LongTensor(num_pos_pad)

        num_order_pad = []
        max_num_order_size = max(num_size_batch)
        for i in range(len(num_order_batch)):
            temp = num_order_batch[i] + [0] * (max_num_order_size - len(num_order_batch[i]))
            num_order_pad.append(temp)
        num_order_pad = torch.LongTensor(num_order_pad)

        num_stack1_batch = copy.deepcopy(num_stack_batch)
        num_stack2_batch = copy.deepcopy(num_stack_batch)
        #num_start2 = output2_lang.n_words - copy_nums - 2
        #unk1 = output1_lang.word2index["UNK"]
        #unk2 = output2_lang.word2index["UNK"]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        # input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
        # input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
        # target1 = torch.LongTensor(target1_batch).transpose(0, 1)
        # target2 = torch.LongTensor(target2_batch).transpose(0, 1)
        input1_var = input1_var.transpose(0, 1)
        input2_var = input2_var.transpose(0, 1)
        target1 = target1.transpose(0, 1)
        target2 = target2.transpose(0, 1)
        parse_graph_pad = torch.LongTensor(parse_graph)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)

        encoder_outputs, encoder_hidden = self.encoder(input1_var, input2_var, input_length, parse_graph_pad)
        copy_num_len = [len(_) for _ in num_pos_batch]
        num_size = max(copy_num_len)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

        decoder_hidden = encoder_hidden[self.n_layers]  # Use last (forward) hidden state from encoder
        if target1 != None:
            all_output1 = self.train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length, batch_size, padding_hidden, seq_mask, num_mask, num_pos_batch, num_order_pad,
                                                 num_stack1_batch)

            all_output2 = self.train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length, batch_size, seq_mask, num_stack2_batch)
            return "train", all_output1, all_output2
        else:
            all_output1 = self.evaluate_tree_double(encoder_outputs, problem_output, num_outputs, batch_size, padding_hidden, seq_mask, num_mask)
            all_output2 = self.evaluate_attn_double(encoder_outputs, decoder_hidden, batch_size, seq_mask)
            if all_output1.score >= all_output2.score:
                return "tree", all_output1.out, all_output1.score
            else:
                return "attn", all_output2.all_output, all_output2.score

    def calculate_loss(self, batch_data):
        input1_var = batch_data['input1']
        input2_var = batch_data['input2']
        input_length = batch_data['input1 len']
        target1 = batch_data['output1']
        target1_length = batch_data['output1 len']
        target2 = batch_data['output2']
        target2_length = batch_data['output2 len']
        num_stack_batch = batch_data['num stack']
        num_size_batch = batch_data['num size']
        generate_list = self.generate_list
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        parse_graph = batch_data['parse graph']
        equ_mask1 = batch_data['equ mask1']
        equ_mask2 = batch_data['equ mask2']
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_list)
        for i in num_size_batch:
            d = i + len(generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        num_pos_pad = []
        max_num_pos_size = max(num_size_batch)
        for i in range(len(num_pos_batch)):
            temp = num_pos_batch[i] + [-1] * (max_num_pos_size - len(num_pos_batch[i]))
            num_pos_pad.append(temp)
        num_pos_pad = torch.LongTensor(num_pos_pad).to(self.device)

        num_order_pad = []
        max_num_order_size = max(num_size_batch)
        for i in range(len(num_order_batch)):
            temp = num_order_batch[i] + [0] * (max_num_order_size - len(num_order_batch[i]))
            num_order_pad.append(temp)
        num_order_pad = torch.LongTensor(num_order_pad).to(self.device)

        num_stack1_batch = copy.deepcopy(num_stack_batch)
        num_stack2_batch = copy.deepcopy(num_stack_batch)
        #num_start2 = output2_lang.n_words - copy_nums - 2
        #unk1 = output1_lang.word2index["UNK"]
        #unk2 = output2_lang.word2index["UNK"]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        # input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
        # input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
        # target1 = torch.LongTensor(target1_batch).transpose(0, 1)
        # target2 = torch.LongTensor(target2_batch).transpose(0, 1)
        # input1_var = input1_var.transpose(0, 1)
        # input2_var = input2_var.transpose(0, 1)
        # target1 = target1.transpose(0, 1)
        # target2 = target2.transpose(0, 1)
        parse_graph_pad = parse_graph.long()

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(input_length)

        encoder_outputs, encoder_hidden = self.encoder(input1_var, input2_var, input_length, parse_graph_pad)
        copy_num_len = [len(_) for _ in num_pos_batch]
        num_size = max(copy_num_len)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

        decoder_hidden = encoder_hidden[:self.n_layers]  # Use last (forward) hidden state from encoder
        all_output1,target1 = self.train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length, batch_size, padding_hidden, seq_mask, num_mask, num_pos_batch, num_order_pad,
                                                num_stack1_batch)

        all_output2,target2_ = self.train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length, batch_size, seq_mask, num_stack2_batch)
        self.loss.reset()
        self.loss.eval_batch(all_output1, target1, equ_mask1)
        self.loss.eval_batch(all_output2, target2_, equ_mask2)
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data):
        input1_var = batch_data['input1']
        input2_var = batch_data['input2']
        input_length = batch_data['input1 len']
        target1 = batch_data['output1']
        target1_length = batch_data['output1 len']
        target2 = batch_data['output2']
        target2_length = batch_data['output2 len']
        num_stack_batch = batch_data['num stack']
        num_size_batch = batch_data['num size']
        generate_list = self.generate_list
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        parse_graph = batch_data['parse graph']
        num_list = batch_data['num list']
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_list)
        for i in num_size_batch:
            d = i + len(generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        num_pos_pad = []
        max_num_pos_size = max(num_size_batch)
        for i in range(len(num_pos_batch)):
            temp = num_pos_batch[i] + [-1] * (max_num_pos_size - len(num_pos_batch[i]))
            num_pos_pad.append(temp)
        num_pos_pad = torch.LongTensor(num_pos_pad).to(self.device)

        num_order_pad = []
        max_num_order_size = max(num_size_batch)
        for i in range(len(num_order_batch)):
            temp = num_order_batch[i] + [0] * (max_num_order_size - len(num_order_batch[i]))
            num_order_pad.append(temp)
        num_order_pad = torch.LongTensor(num_order_pad).to(self.device)

        num_stack1_batch = copy.deepcopy(num_stack_batch)
        num_stack2_batch = copy.deepcopy(num_stack_batch)
        #num_start2 = output2_lang.n_words - copy_nums - 2
        #unk1 = output1_lang.word2index["UNK"]
        #unk2 = output2_lang.word2index["UNK"]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        # input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
        # input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
        # target1 = torch.LongTensor(target1_batch).transpose(0, 1)
        # target2 = torch.LongTensor(target2_batch).transpose(0, 1)
        # input1_var = input1_var.transpose(0, 1)
        # input2_var = input2_var.transpose(0, 1)
        # target1 = target1.transpose(0, 1)
        # target2 = target2.transpose(0, 1)
        parse_graph_pad = parse_graph.long()

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(input_length)

        encoder_outputs, encoder_hidden = self.encoder(input1_var, input2_var, input_length, parse_graph_pad)
        copy_num_len = [len(_) for _ in num_pos_batch]
        num_size = max(copy_num_len)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

        decoder_hidden = encoder_hidden[:self.n_layers]  # Use last (forward) hidden state from encoder
        all_output1 = self.evaluate_tree_double(encoder_outputs, problem_output, num_outputs, batch_size, padding_hidden, seq_mask, num_mask)
        all_output2 = self.evaluate_attn_double(encoder_outputs, decoder_hidden, batch_size, seq_mask)
        if all_output1.score >= all_output2.score:
            output1=self.convert_idx2symbol1(all_output1.out,num_list[0],copy_list(num_stack1_batch[0]))
            targets1=self.convert_idx2symbol1(target1[0],num_list[0],copy_list(num_stack1_batch[0]))
            return "tree", output1, targets1
        else:
            output2=self.convert_idx2symbol2(torch.tensor(all_output2.all_output).view(1,-1),num_list,copy_list(num_stack2_batch))
            targets2=self.convert_idx2symbol2(target2,num_list,copy_list(num_stack2_batch))
            return "attn", output2, targets2

    def train_tree_double(self, encoder_outputs, problem_output, all_nums_encoder_outputs, target, target_length, batch_size, padding_hidden, seq_mask, num_mask, num_pos, num_order_pad,
                          nums_stack_batch):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[:,t].tolist(), outputs, nums_stack_batch)
            target[:,t] = target_t
            # if USE_CUDA:
            #     generate_input = generate_input.cuda()
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:,t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < self.num_start1:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - self.num_start1].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        
        return all_node_outputs,target

    def train_attn_double(self, encoder_outputs, decoder_hidden, target, target_length, batch_size, seq_mask, nums_stack_batch):
        max_target_length = max(target_length)
        decoder_input = torch.LongTensor([self.sos2] * batch_size).to(self.device)
        all_decoder_outputs = torch.zeros(batch_size, max_target_length, self.output2_size).to(self.device)
        #all_decoder_outputs = []

        seq_mask = torch.unsqueeze(seq_mask,dim=1)

        # Move new Variables to CUDA
        # if USE_CUDA:
        #     all_decoder_outputs = all_decoder_outputs.cuda()
        if random.random() < self.teacher_force_ratio:
            # if random.random() < 0:
            # Run through decoder one time step at a time
            #decoder_inputs = torch.cat([decoder_input.view(batch_size,1),target],dim=1)[:,:-1]
            all_decoder_outputs = []
            for t in range(max_target_length):
                #decoder_inputs[:,t]=decoder_input
                #decoder_input = decoder_inputs[:,t]
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask.squeeze(1))
                #all_decoder_outputs[:,t,:] = decoder_output
                all_decoder_outputs.append(decoder_output)
                decoder_input = self.generate_decoder_input(target[:,t].tolist(), decoder_output, nums_stack_batch)
                target[:,t] = decoder_input
            all_decoder_outputs = torch.stack(all_decoder_outputs,dim=1)
        else:
            decoder_input = torch.LongTensor([self.sos2] * batch_size).to(self.device)
            beam_list = list()
            score = torch.zeros(batch_size).to(self.device)
            # if USE_CUDA:
            #     score = score.cuda()
            beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                beam_len = len(beam_list)
                beam_scores = torch.zeros(batch_size, self.output2_size * beam_len).to(self.device)
                all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2)).to(self.device)
                all_outputs = torch.zeros(batch_size * beam_len, max_target_length, self.output2_size).to(self.device)
                
                # if USE_CUDA:
                #     beam_scores = beam_scores.cuda()
                #     all_hidden = all_hidden.cuda()
                #     all_outputs = all_outputs.cuda()

                for b_idx in range(len(beam_list)):
                    decoder_input = beam_list[b_idx].input_var
                    decoder_hidden = beam_list[b_idx].hidden

                    decoder_input = decoder_input.to(self.device)
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs, seq_mask.squeeze(1))

                    score = F.log_softmax(decoder_output, dim=1)
                    beam_score = beam_list[b_idx].score
                    beam_score = beam_score.unsqueeze(1)
                    repeat_dims = [1] * beam_score.dim()
                    repeat_dims[1] = score.size(1)
                    beam_score = beam_score.repeat(*repeat_dims)
                    score = score + beam_score
                    beam_scores[:, b_idx * self.output2_size:(b_idx + 1) * self.output2_size] = score
                    all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                    beam_list[b_idx].all_output[:,t,:] = decoder_output
                    
                    all_outputs[batch_size * b_idx: batch_size * (b_idx + 1),:, :] = \
                        beam_list[b_idx].all_output

                topv, topi = beam_scores.topk(self.beam_size, dim=1)
                beam_list = list()

                for k in range(self.beam_size):
                    temp_topk = topi[:, k]
                    temp_input = temp_topk % self.output2_size
                    temp_input = temp_input.data
                    temp_beam_pos = temp_topk // self.output2_size

                    indices = torch.LongTensor(range(batch_size)).to(self.device)
                    indices += temp_beam_pos * batch_size

                    temp_hidden = all_hidden.index_select(dim=1, index=indices)
                    temp_output = all_outputs.index_select(dim=0, index=indices)

                    beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))

            all_decoder_outputs = beam_list[0].all_output
            for t in range(max_target_length):
                target[:,t] = self.generate_decoder_input(
                    target[:,t].tolist(), all_decoder_outputs[:,t], nums_stack_batch)
        return all_decoder_outputs,target

    def evaluate_tree_double(self, encoder_outputs, problem_output, all_nums_encoder_outputs, batch_size, padding_hidden, seq_mask, num_mask):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        #num_start = output_lang.num_start
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(self.max_out_len):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                                                                                                           seq_mask, num_mask)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                topv, topi = out_score.topk(self.beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < self.num_start1:
                        generate_input = torch.LongTensor([out_token]).to(self.device)
                        # if USE_CUDA:
                        #     generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - self.num_start1].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:self.beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0]

    def evaluate_attn_double(self, encoder_outputs, decoder_hidden, batch_size, seq_mask):
        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([self.sos2]).to(self.device)  # SOS
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(self.max_out_len):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == self.eos2:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0]
            beam_scores = torch.zeros(self.output2_size * beam_len).to(self.device)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2).to(self.device)
            
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == self.eos2:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                #decoder_output = self.out(decoder_output).squeeze(dim=1)
                score = F.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * self.output2_size:(current_idx + 1) * self.output2_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(self.beam_size)

            for k in range(self.beam_size):
                word_n = int(topi[k])
                word_input = word_n % self.output2_size
                temp_input = torch.LongTensor([word_input]).to(self.device)
                indices = int(word_n / self.output2_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices] + [word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < self.beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:self.beam_size]
        return beam_list[0]
    
    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                if i == -1:
                    indices.append(0)
                    masked_index.append(temp_1)
                    continue
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices = indices + [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index = masked_index + [temp_1 for _ in range(len(num_pos[b]), num_size)]
        # indices = torch.LongTensor(indices)
        # masked_index = torch.ByteTensor(masked_index)
        indices = torch.LongTensor(indices).to(self.device)
        masked_index = torch.BoolTensor(masked_index).to(self.device)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index

    def generate_tree_input(self, target, decoder_output, nums_stack_batch):
        # when the decoder input is copied num but the num has two pos, chose the max
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == self.unk1:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, self.num_start1 + num] > max_score:
                        target[i] = num + self.num_start1
                        max_score = decoder_output[i, self.num_start1 + num]
            if target_input[i] >= self.num_start1:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def generate_decoder_input(self, target, decoder_output, nums_stack_batch):
        # when the decoder input is copied num but the num has two pos, chose the max
        # if USE_CUDA:
        #     decoder_output = decoder_output.cpu()
        target=torch.LongTensor(target).to(self.device)
        for i in range(target.size(0)):
            if target[i] == self.unk2:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, self.num_start2 + num] > max_score:
                        target[i] = num + self.num_start2
                        max_score = decoder_output[i, self.num_start2 + num]
        return target

    def convert_idx2symbol1(self, output, num_list, num_stack):
        #batch_size=output.size(0)
        '''batch_size=1'''
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token1, self.out_eos_token1, self.out_pad_token1]:
                break
            symbol = self.out_idx2symbol1[idx]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list

    def convert_idx2symbol2(self, output, num_list, num_stack):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token2, self.out_eos_token2, self.out_pad_token2]:
                    break
                symbol = self.out_idx2symbol2[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                elif symbol == SpecialTokens.UNK_TOKEN:
                    try:
                        pos_list = num_stack[b_i].pop()
                        c = num_list[b_i][pos_list[0]]
                        res.append(c)
                    except:
                        res.append(symbol)
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    # def get_all_number_encoder_outputs(self,encoder_outputs, num_pos, num_size, hidden_size):
    #     indices = list()
    #     sen_len = encoder_outputs.size(1)
    #     batch_size=encoder_outputs.size(0)
    #     masked_index = []
    #     temp_1 = [1 for _ in range(hidden_size)]
    #     temp_0 = [0 for _ in range(hidden_size)]
    #     for b in range(batch_size):
    #         for i in num_pos[b]:
    #             indices.append(i + b * sen_len)
    #             masked_index.append(temp_0)
    #         indices += [0 for _ in range(len(num_pos[b]), num_size)]
    #         masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    #     indices = torch.LongTensor(indices).to(self.device)
    #     masked_index = torch.BoolTensor(masked_index).to(self.device)

    #     masked_index = masked_index.view(batch_size, num_size, hidden_size)
    #     all_outputs = encoder_outputs.contiguous()
    #     all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    #     all_num = all_embedding.index_select(0, indices)
    #     all_num = all_num.view(batch_size, num_size, hidden_size)
    #     return all_num.masked_fill_(masked_index, 0.0)

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class AttnDecoderRNN_(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN_, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
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

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)

def train_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs, target, target_length,
                      num_start,batch_size, padding_hidden, seq_mask, 
                      num_mask, num_pos, num_order_pad, nums_stack_batch, unk, 
                      encoder, numencoder, predict, generate, merge):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []
    
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    return loss  # , loss_0.item(), loss_1.item()


def train_attn_double(encoder_outputs, decoder_hidden, target, target_length,
                      sos, batch_size, seq_mask, 
                      num_start, nums_stack_batch, unk,
                      decoder, beam_size, use_teacher_forcing):
    # Prepare input and output variables
    decoder_input = torch.LongTensor([sos] * batch_size)

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

            #                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
            #                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    #                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                #                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                score = F.log_softmax(decoder_output, dim=1)
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )
    
    return loss


def train_double(input1_batch, input2_batch, input_length, target1_batch, target1_length, target2_batch, target2_length, 
                 num_stack_batch, num_size_batch, generate_num1_ids, 
                 encoder, numencoder, predict, generate, merge, decoder, unk1, unk2, num_start1, num_start2, sos2, 
                 num_pos_batch, num_order_batch, parse_graph_batch, beam_size=5, use_teacher_forcing=0.83, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_num1_ids)
    for i in num_size_batch:
        d = i + len(generate_num1_ids)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    
    num_pos_pad = []
    max_num_pos_size = max(num_size_batch)
    for i in range(len(num_pos_batch)):
        temp = num_pos_batch[i] + [-1] * (max_num_pos_size-len(num_pos_batch[i]))
        num_pos_pad.append(temp)
    num_pos_pad = torch.LongTensor(num_pos_pad)

    num_order_pad = []
    max_num_order_size = max(num_size_batch)
    for i in range(len(num_order_batch)):
        temp = num_order_batch[i] + [0] * (max_num_order_size-len(num_order_batch[i]))
        num_order_pad.append(temp)
    num_order_pad = torch.LongTensor(num_order_pad)
    
    num_stack1_batch = copy.deepcopy(num_stack_batch)
    num_stack2_batch = copy.deepcopy(num_stack_batch)
    
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
    # input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
    # target1 = torch.LongTensor(target1_batch).transpose(0, 1)
    # target2 = torch.LongTensor(target2_batch).transpose(0, 1)
    # parse_graph_pad = torch.LongTensor(parse_graph_batch)
    input1_var = input1_batch.transpose(0, 1)
    input2_var = input2_batch.transpose(0, 1)
    target1 = target1_batch.transpose(0, 1)
    target2 = target2_batch.transpose(0, 1)
    parse_graph_pad = parse_graph_batch

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    numencoder.train()
    predict.train()
    generate.train()
    merge.train()
    decoder.train()

    if USE_CUDA:
        input1_var = input1_var.cuda()
        input2_var = input2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        num_pos_pad = num_pos_pad.cuda()
        num_order_pad = num_order_pad.cuda()
        parse_graph_pad = parse_graph_pad.cuda()

    # Run words through encoder
    
    encoder_outputs, encoder_hidden = encoder(input1_var, input2_var, input_length, parse_graph_pad)
    copy_num_len = [len(_) for _ in num_pos_batch]
    num_size = max(copy_num_len)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, 
                                                                       batch_size, num_size, encoder.hidden_size)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs, 
                                                              num_pos_pad, num_order_pad)
    num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    loss_0 = train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length,
                               num_start1, batch_size, padding_hidden, seq_mask, 
                               num_mask, num_pos_batch, num_order_pad, num_stack1_batch, unk1, 
                               encoder, numencoder, predict, generate, merge)
    
    loss_1 = train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length,
                               sos2, batch_size, seq_mask, 
                               num_start2, num_stack2_batch, unk2, 
                               decoder, beam_size, use_teacher_forcing)
    
    loss = loss_0 + loss_1
    loss.backward()
    
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs, 
                         num_start, batch_size, padding_hidden, seq_mask, num_mask, 
                         max_length, num_pos, num_order_pad, 
                         encoder, numencoder, predict, generate, merge, beam_size):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]


def evaluate_attn_double(encoder_outputs, decoder_hidden, 
                         sos,eos, batch_size, seq_mask, max_length, 
                         decoder, beam_size):
    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([sos])  # SOS
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == eos:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0]
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == eos:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = F.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0]


def evaluate_double(input1_batch, input2_batch, input_length, generate_num1_ids, num_start1,sos2, eos2, 
                    encoder, numencoder, predict, generate, merge, decoder, num_pos_batch, num_order_batch, parse_graph_batch, 
                    beam_size=5, english=False, max_length=30):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # num_pos_pad = torch.LongTensor([num_pos_batch])
    # num_order_pad = torch.LongTensor([num_order_batch])
    # parse_graph_pad = torch.LongTensor(parse_graph_batch)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input1_var = torch.LongTensor(input1_batch).transpose()
    # input2_var = torch.LongTensor(input2_batch).unsqueeze(1)
    num_pos_pad = torch.LongTensor(num_pos_batch)
    num_order_pad = torch.LongTensor(num_order_batch)
    parse_graph_pad = parse_graph_batch
    input1_var = input1_batch.transpose(0,1)
    input2_var = input2_batch.transpose(0,1)

    num_mask = torch.ByteTensor(1, len(num_pos_batch[0]) + len(generate_num1_ids)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    numencoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    decoder.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input1_var = input1_var.cuda()
        input2_var = input2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        num_pos_pad = num_pos_pad.cuda()
        num_order_pad = num_order_pad.cuda()
        parse_graph_pad = parse_graph_pad.cuda()
    # Run words through encoder

    encoder_outputs, encoder_hidden = encoder(input1_var, input2_var, input_length, parse_graph_pad)
    copy_num_len = [len(_) for _ in num_pos_batch]
    num_size = max(copy_num_len)
    #num_size = len(num_pos_batch)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, batch_size, 
                                                                        num_size, encoder.hidden_size)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs, 
                                                               num_pos_pad, num_order_pad)
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    
    tree_beam = evaluate_tree_double(encoder_outputs, problem_output, num_outputs,
                                     num_start1, batch_size, padding_hidden, seq_mask, num_mask, 
                                     max_length, num_pos_batch, num_order_pad, 
                                     encoder, numencoder, predict, generate, merge, beam_size)
    
    attn_beam = evaluate_attn_double(encoder_outputs, decoder_hidden,sos2,eos2,
                                     batch_size, seq_mask, max_length, 
                                     decoder, beam_size)
    
    if tree_beam.score >= attn_beam.score:
        return "tree", tree_beam.out, tree_beam.score
    else:
        return "attn", attn_beam.all_output, attn_beam.score

def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target

def masked_cross_entropy(logits, target, length):
    if isinstance(length,torch.Tensor):
        if torch.cuda.is_available():
            length = length.cuda()
    else:
        if torch.cuda.is_available():
            length = torch.LongTensor(length).cuda()
        else:
            length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def replace_masked_values(tensor, mask, replace_with):
    return tensor.masked_fill((1 - mask).bool(), replace_with)

