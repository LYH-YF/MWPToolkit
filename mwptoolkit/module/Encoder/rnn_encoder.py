from copy import deepcopy

import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import SeqAttention
from mwptoolkit.module.Attention.group_attention import GroupAttention
from mwptoolkit.module.Attention.hierarchical_attention import Attention
from mwptoolkit.module.Attention.seq_attention import Attention as Attention_x
from mwptoolkit.module.Embedder.position_embedder import PositionalEncoding
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.transformer_encoder import GroupATTEncoder
from mwptoolkit.module.Layer.transformer_layer import PositionwiseFeedForward,GAEncoderLayer


class BasicRNNEncoder(nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) encoder.
    """
    def __init__(self, embedding_size, hidden_size, num_layers, rnn_cell_type, dropout_ratio, bidirectional=True,batch_first=True):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.batch_first = batch_first

        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        if self.batch_first:
            batch_size = input_embeddings.size(0)
        else:
            batch_size = input_embeddings.size(1)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            tp_vec = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            return tp_vec.to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing encoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length, batch_first=self.batch_first, enforce_sorted=True)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)

        return outputs, hidden_states


class SelfAttentionRNNEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, context_size, num_layers, rnn_cell_type, dropout_ratio, bidirectional=True):
        super(SelfAttentionRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention = SeqAttention(hidden_size, context_size)

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            Torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            tp_vec = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            return tp_vec.to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing encoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        r""" Implement the encoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (Torch.Tensor): length of input sequence, shape: [batch_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length, batch_first=True, enforce_sorted=True)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.bidirectional:
            encoder_outputs = outputs[:, :, self.hidden_size:] + outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                encoder_hidden = (hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous())
            else:
                encoder_hidden = hidden_states[::2].contiguous()

        outputs, attn = self.attention.forward(encoder_outputs, encoder_outputs, mask=None)

        return outputs, hidden_states


class GroupAttentionRNNEncoder(nn.Module):
    def __init__(self, emb_size=100, hidden_size=128, n_layers=1, bidirectional=False, \
                 rnn_cell=None, rnn_cell_name='gru', variable_lengths=True, \
                 d_ff=2048, dropout=0.3, N=1):
        super(GroupAttentionRNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.dropout = dropout

        if bidirectional:
            self.d_model = 2*hidden_size
        else:
            self.d_model = hidden_size
        ff = PositionwiseFeedForward(self.d_model, d_ff, dropout)

        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU

        if rnn_cell is None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=self.dropout)
        else:
            self.rnn = rnn_cell
        self.group_attention = GroupAttention(8, self.d_model)
        self.onelayer = GroupATTEncoder(GAEncoderLayer(self.d_model, deepcopy(self.group_attention), deepcopy(ff), dropout), N)

    def forward(self, embedded, input_var, split_list, input_lengths=None):
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True,\
                                                        enforce_sorted=True)
        else:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True,\
                                                        enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        src_mask = self.group_attention.get_mask(input_var, split_list)
        output = self.onelayer(output, src_mask)
        return output, hidden



class HWCPEncoder(nn.Module):
    r"""Hierarchical word-clause-problem encoder"""
    def __init__(self,embedding_model,embedding_size, hidden_size=512, span_size=0, dropout_ratio=0.4):
        super(HWCPEncoder, self).__init__()
        self.hidden_size = hidden_size
        

        self.embedding = embedding_model
        # word encoding
        self.word_rnn = nn.GRU(embedding_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        # span encoding
        # span sequence
        self.span_attn = Attention(self.hidden_size, mix=True, fn=True)
        self.pos_enc = PositionalEncoding(span_size, hidden_size)
        # merge subtree/word node
        self.to_parent = Attention(self.hidden_size, mix=True, fn=True)
        return

    def forward(self, input_var, input_lengths, span_length, tree=None):
        device = span_length.device

        word_outputs = []
        span_inputs = []

        input_vars = input_var
        trees = tree
        bi_word_hidden = None
        for span_index, input_var in enumerate(input_vars):
            input_length = input_lengths[span_index]

            # word encoding
            embedded = self.embedding(input_var)
            
            # word level encoding
            word_output, bi_word_hidden = self.word_level_forward(embedded, input_length, bi_word_hidden)
            word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)

            # tree encoding/clause level
            tree_batch = trees[span_index]
            span_span_input = self.clause_level_forward(word_output, tree_batch)
            span_input = torch.cat(span_span_input, dim=0)

            span_inputs.append(span_input.unsqueeze(1))
            word_outputs.append(word_output)

        # span encoding / problem level
        span_input = torch.cat(span_inputs, dim=1)
        span_mask = self.get_mask(span_length, span_input.size(1))
        span_output, _ = self.problem_level_forword(span_input, span_mask)
        span_output = span_output * (span_mask == 0).unsqueeze(-1)
        dim0 = torch.arange(span_output.size(0)).to(device)
        span_hidden = span_output[dim0, span_length - 1].unsqueeze(0)

        return (span_output, word_outputs), span_hidden #【4，5,512】5*【4，length,512】【1,4,512】

    def word_level_forward(self, embedding_inputs, input_length, bi_word_hidden=None):
        # at least 1 word in some full padding span
        pad_input_length = input_length.clone()
        pad_input_length[pad_input_length == 0] = 1
        embedded = nn.utils.rnn.pack_padded_sequence(embedding_inputs, pad_input_length, batch_first=True, enforce_sorted=False)
        word_output, bi_word_hidden = self.word_rnn(embedded, bi_word_hidden)
        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)
        #word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)
        return word_output, bi_word_hidden

    def clause_level_forward(self, word_output, tree_batch):
        device = word_output.device
        span_span_input = []
        for b_i, data_word_output in enumerate(word_output):
            data_word_output = data_word_output.unsqueeze(0)
            tree = tree_batch[b_i]
            if tree is not None:
                data_span_input = self.dependency_encode(data_word_output, tree.root)
            else:
                pad_hidden = torch.zeros(1, self.hidden_size).to(device)
                data_span_input = pad_hidden
            span_span_input.append(data_span_input)
        return span_span_input

    def problem_level_forword(self, span_input, span_mask):
        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, span_attn = self.span_attn(span_output, span_output, span_mask)
        return span_output, span_attn

    def bi_combine(self, output, hidden):
        # combine forward and backward LSTM
        # (num_layers * num_directions, batch, hidden_size).view(num_layers, num_directions, batch, hidden_size)
        hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    def dependency_encode(self, word_output, node):
        pos = node.position
        word_vector = word_output[:, pos]
        if node.is_leaf:
            vector = word_vector
        else:
            children = node.left_nodes + node.right_nodes
            children_vector = [self.dependency_encode(word_output, child).unsqueeze(1) for child in children]
            children_vector = torch.cat(children_vector, dim=1)
            query = word_vector.unsqueeze(1)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        return vector

    def get_mask(self, encode_lengths, pad_length):
        device = encode_lengths.device
        batch_size = encode_lengths.size(0)
        index = torch.arange(pad_length).to(device)

        mask = (index.unsqueeze(0).expand(batch_size, -1) >= encode_lengths.unsqueeze(-1)).byte()
        # save one position for full padding span to prevent nan in softmax
        # invalid value in full padding span will be ignored in span level attention
        mask[mask.sum(dim=-1) == pad_length, 0] = 0
        return mask

class SalignedEncoder(nn.Module):
    """ Simple RNN encoder with attention which also extract variable embedding.

    Args:
        dim_embed (int): Dimension of input embedding.
        dim_hidden (int): Dimension of encoder RNN.
        dim_last (int): Dimension of the last state will be transformed to.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, dim_embed, dim_hidden, dim_last, dropout_rate,
                 dim_attn_hidden=256):
        super(SalignedEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embed,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())
        self.attn = Attention_x(dim_hidden * 2, dim_hidden * 2,
                              dim_attn_hidden)
        # self.embedding_one = torch.nn.Parameter(
        #     torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        # self.embedding_pi = torch.nn.Parameter(
        #     torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.register_buffer('padding',
                             torch.zeros(dim_hidden * 2))
        self.embeddings = torch.nn.Parameter(
            torch.normal(torch.zeros(20, 2 * dim_hidden), 0.01))
        self.dim_hidden = dim_hidden

    def initialize_fix_constant(self, con_len, device):
        self.embedding_con = [torch.nn.Parameter(
            torch.normal(torch.zeros(2 * self.dim_hidden), 0.01)).to(device) for c in range(con_len)]
    def get_fix_constant(self):
        return self.embedding_con

    def forward(self, inputs, lengths, constant_indices):
        """

        Args:
            inputs (tensor): Indices of words. The shape is `B x T x 1`.
            length (list of int): Length of inputs.
            constant_indices (list of int): Each list contains list

        Return:
            outputs (tensor): Encoded sequence. The shape is
                `B x T x dim_hidden`.
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True)
        hidden_state = None
        outputs, hidden_state = self.rnn(packed, hidden_state)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                            batch_first=True)

        # reshape (2, batch, dim_hidden) to (batch, dim_hidden)
        hidden_state = \
            (hidden_state[0].transpose(1, 0).contiguous()
             .view(hidden_state[0].size(1), -1),
             hidden_state[1].transpose(1, 0).contiguous()
             .view(hidden_state[1].size(1), -1))
        hidden_state = \
            (self.mlp1(hidden_state[0]).unsqueeze(0),
             self.mlp2(hidden_state[1]).unsqueeze(0))

        batch_size = outputs.size(0)
        # operands = [self.embedding_con + #[self.embedding_one, self.embedding_pi] +
        #             [outputs[b][i]
        #              for i in constant_indices[b]]
        #             for b in range(batch_size)]
        operands = [[outputs[b][i] for i in constant_indices[b]] for b in range(batch_size)]
        # operands = [[self.embedding_one, self.embedding_pi] +
        #             [self.embeddings[i]
        #              for i in range(len(constant_indices[b]))]
        #             for b in range(batch_size)]
        # n_operands, operands = pad_and_cat(operands, self.padding)

        # attns = []
        # for i in range(operands.size(1)):
        #     attn = self.attn(outputs, operands[:, i], lengths)
        #     attns.append(attn)

        # operands = [[self.embedding_one, self.embedding_pi]
        #             + [attns[i][b]
        #                for i in range(len(constant_indices[b]))]
        #             for b in range(batch_size)]

        return outputs, hidden_state, operands

# class SalignedEncoder(torch.nn.Module):
#     """ Simple RNN encoder.

#     Args:
#         dim_embed (int): Dimension of input embedding.
#         dim_hidden (int): Dimension of encoder RNN.
#         dim_last (int): Dimension of the last state will be transformed to.
#         dropout_rate (float): Dropout rate.
#     """
#     def __init__(self, dim_embed, dim_hidden, dim_last, dropout_rate):
#         super(SalignedEncoder, self).__init__()
#         self.rnn = torch.nn.LSTM(dim_embed,
#                                  dim_hidden,
#                                  1,
#                                  bidirectional=True,
#                                  batch_first=True)
#         self.mlp1 = torch.nn.Sequential(
#             torch.nn.Linear(dim_hidden * 2, dim_last),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Tanh())
#         self.mlp2 = torch.nn.Sequential(
#             torch.nn.Linear(dim_hidden * 2, dim_last),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Tanh())

#     def forward(self, inputs, lengths):
#         """

#         Args:
#             inputs (tensor): Indices of words. The shape is `B x T x 1`.
#             length (list of int): Length of inputs.

#         Return:
#             outputs (tensor): Encoded sequence. The shape is
#                 `B x T x dim_hidden`.
#         """
#         packed = torch.nn.utils.rnn.pack_padded_sequence(
#             inputs, lengths, batch_first=True)
#         hidden_state = None
#         outputs, hidden_state = self.rnn(packed, hidden_state)
#         outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
#                                                             batch_first=True)

#         # reshape (2, batch, dim_hidden) to (batch, dim_hidden)
#         hidden_state = \
#             (hidden_state[0].transpose(1, 0).contiguous()
#              .view(hidden_state[0].size(1), -1),
#              hidden_state[1].transpose(1, 0).contiguous()
#              .view(hidden_state[1].size(1), -1))
#         hidden_state = \
#             (self.mlp1(hidden_state[0]).unsqueeze(0),
#              self.mlp2(hidden_state[1]).unsqueeze(0))

#         return outputs, hidden_state


# class GroupAttentionRNNEncoder(nn.Module):
#     def __init__(self, embedding_size, hidden_size, num_layers, bidirectional, rnn_cell_type, dropout_ratio, in_word2idx, d_ff=2048, N=1):
#         super(GroupAttentionRNNEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.rnn_cell_type = rnn_cell_type
#         self.num_layers = num_layers
#         self.num_directions = 2 if self.bidirectional else 1
#         if bidirectional:
#             self.d_model = 2 * hidden_size
#         else:
#             self.d_model = hidden_size
#         if rnn_cell_type == 'lstm':
#             self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
#         elif rnn_cell_type == 'gru':
#             self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
#         elif rnn_cell_type == 'rnn':
#             self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
#         else:
#             raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")
#         #self.group_attention = GroupAttention(8, self.d_model, dropout_ratio, in_word2idx)
#         self.onelayer = GroupATTEncoder(self.d_model, 8, self.d_model, dropout_ratio, d_ff, N)
#         self.separate_list=[]
#         # chinese dataset
#         try:
#             self.separate_list.append(in_word2idx['．'])
#         except:
#             pass
#         try:
#             self.separate_list.append(in_word2idx["，"])
#         except:
#             pass
#         # english dataset
#         try:
#             self.separate_list.append(in_word2idx["."])
#         except:
#             pass
#         try:
#             self.separate_list.append(in_word2idx[","])
#         except:
#             pass

#     def init_hidden(self, input_embeddings):
#         r""" Initialize initial hidden states of RNN.

#         Args:
#             input_embeddings (Torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

#         Returns:
#             Torch.Tensor: the initial hidden states.
#         """
#         batch_size = input_embeddings.size(0)
#         device = input_embeddings.device
#         if self.rnn_cell_type == 'lstm':
#             h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
#             c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
#             hidden_states = (h_0, c_0)
#             return hidden_states
#         elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
#             tp_vec = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
#             return tp_vec.to(device)
#         else:
#             raise NotImplementedError("No such rnn type {} for initializing encoder states.".format(self.rnn_type))

#     def forward(self, input_seq, input_embeddings, input_length, hidden_states=None):
#         if hidden_states is None:
#             hidden_states = self.init_hidden(input_embeddings)

#         packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length, batch_first=True, enforce_sorted=False)
#         outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

#         outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#         #src_mask = self.group_attention.get_mask(input_seq)
#         src_mask = self.get_mask(input_seq)
#         outputs = self.onelayer(outputs, src_mask)
#         return outputs, hidden_states
    
#     def src_to_mask(self,src):
#         src = src.cpu().numpy()
#         batch_data_mask_tok = []
#         for encode_sen_idx in src:

#             token = 1
#             mask = [0] * len(encode_sen_idx)
#             for num in range(len(encode_sen_idx)):
#                 mask[num] = token
#                 # if (encode_sen_idx[num] == self.in_word2idx['．'] or encode_sen_idx[num] == self.in_word2idx["，"]) \
#                 #         and num != len(encode_sen_idx) - 1:
#                 if (encode_sen_idx[num] in self.separate_list) and num != len(encode_sen_idx) - 1:
#                     token += 1
#                 if encode_sen_idx[num]==0:mask[num] = 0
#             for num in range(len(encode_sen_idx)):
#                 if mask[num] == token and token != 1:
#                     mask[num] = 1000
#             batch_data_mask_tok.append(mask)
#         return torch.tensor(batch_data_mask_tok)

#     def get_mask(self,src,pad=0):
#         device=src.device
#         mask = self.src_to_mask(src)
#         self.src_mask_self = self.group_mask(mask,"self",pad).bool().unsqueeze(1)
#         self.src_mask_between = self.group_mask(mask,"between",pad).bool().unsqueeze(1)
#         self.src_mask_question = self.group_mask(mask, "question", pad).bool().unsqueeze(1)
#         self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
#         self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
#         self.final = torch.cat((self.src_mask_between.to(device),self.src_mask_self.to(device),self.src_mask_global.to(device),self.src_mask_question.to(device)),1)
#         return self.final.to(device)
    
#     def group_mask(self,batch,type="self",pad=0):
#         length = batch.shape[1]
#         lis = []
#         if type=="self":
#             for tok in batch:
#                 mask = torch.zeros(tok.shape)
#                 mask = torch.unsqueeze(mask,-1)
#                 for ele in tok:
#                     if ele == pad:
#                         copy = torch.zeros(length)
#                     else:
#                         copy = torch.clone(tok)
#                         if ele != 1000:copy[copy == 1000] = 0
#                         copy[copy != ele] = 0
#                         copy[copy == ele] = 1
#                         #print("self copy",copy)
#                     copy = torch.unsqueeze(copy,-1)
#                     mask = torch.cat([mask,copy.float()],dim=1)
#                 mask = mask[:,1:]
#                 mask = mask.transpose(0,1)
#                 #mask = np.expand_dims(mask,0)
#                 mask = torch.unsqueeze(mask,0)
#                 lis.append(mask)
#             #res = np.concatenate(tuple(lis))
#             res = torch.cat(lis)
#         elif type=="between":
#             for tok in batch:
#                 # mask = np.zeros(tok.shape)
#                 # mask = np.expand_dims(mask,-1)
#                 mask = torch.zeros(tok.shape)
#                 mask = torch.unsqueeze(mask,-1)
#                 for ele in tok:
#                     if ele == pad:
#                         copy = torch.zeros(length)
#                         #copy = np.zeros(length)
#                     else:
#                         copy = torch.clone(tok)
#                         copy[copy==1000] = 0
#                         copy[copy ==ele] = 0
#                         copy[copy!= 0] = 1
#                         '''
#                         copy[copy != ele and copy != 1000] = 1
#                         copy[copy == ele or copy == 1000] = 0
#                         '''
#                     # copy = np.expand_dims(copy,-1)
#                     # mask = np.concatenate((mask,copy),axis=1)
#                     copy = torch.unsqueeze(copy,-1)
#                     mask = torch.cat([mask,copy.float()],dim=1)
#                 mask = mask[:,1:]
#                 mask = mask.transpose(0,1)
#                 #mask = np.expand_dims(mask,0)
#                 mask = torch.unsqueeze(mask,0)
#                 lis.append(mask)
#             #res = np.concatenate(tuple(lis))
#             res = torch.cat(lis)
#         elif type == "question":
#             for tok in batch:
#                 # mask = np.zeros(tok.shape)
#                 # mask = np.expand_dims(mask,-1)
#                 mask = torch.zeros(tok.shape)
#                 mask = torch.unsqueeze(mask,-1)
#                 for ele in tok:
#                     if ele == pad:
#                         #copy = np.zeros(length)
#                         copy = torch.zeros(length)
#                     else:
#                         copy = torch.clone(tok)
#                         copy[copy != 1000] = 0
#                         copy[copy == 1000] = 1
#                     if ele==1000:
#                         copy[copy==0] = -1
#                         copy[copy==1] = 0
#                         copy[copy==-1] = 1
#                     # copy = np.expand_dims(copy,-1)
#                     # mask = np.concatenate((mask,copy),axis=1)
#                     copy = torch.unsqueeze(copy,-1)
#                     mask = torch.cat([mask,copy.float()],dim=1)
#                 mask = mask[:,1:]
#                 mask = mask.transpose(0,1)
#                 #mask = np.expand_dims(mask,0)
#                 mask = torch.unsqueeze(mask,0)
#                 lis.append(mask)
#             #res = np.concatenate(tuple(lis))
#             res = torch.cat(lis)
#         else:
#             return "error"
#         return res
