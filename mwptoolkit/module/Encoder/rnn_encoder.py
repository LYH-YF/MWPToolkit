import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import SeqAttention
from mwptoolkit.module.Attention.group_attention import GroupAttention
from mwptoolkit.module.Attention.hierarchical_attention import Attention
from mwptoolkit.module.Embedder.position_embedder import PositionalEncoding
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.transformer_encoder import GroupATTEncoder
#from mwptoolkit.module.Layer.transformer_layer import Encoder,EncoderLayer,PositionwiseFeedForward


class BasicRNNEncoder(nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) encoder.
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 rnn_cell_type,
                 dropout_ratio,
                 bidirectional=True):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

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

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length,
                                                                          batch_first=True, enforce_sorted=False)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden_states

class SelfAttentionRNNEncoder(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 context_size,
                 num_layers,
                 rnn_cell_type,
                 dropout_ratio,
                 bidirectional=True):
        super(SelfAttentionRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention=SeqAttention(hidden_size,context_size)

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

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length,
                                                                          batch_first=True, enforce_sorted=False)

        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.bidirectional:
            encoder_outputs = outputs[:, :, self.hidden_size:] + outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                encoder_hidden = (hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous())
            else:
                encoder_hidden = hidden_states[::2].contiguous()

        outputs, attn = self.attention.forward(encoder_outputs,encoder_outputs,mask=None)

        return outputs, hidden_states


class GroupAttentionRNNEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size,num_layers, bidirectional, rnn_cell_type, dropout_ratio,in_word2idx,d_ff=2048,N=1):
        super(GroupAttentionRNNEncoder, self).__init__()
        self.hidden_size=hidden_size
        self.bidirectional = bidirectional
        self.rnn_cell_type=rnn_cell_type
        self.num_layers=num_layers
        self.num_directions = 2 if self.bidirectional else 1
        if bidirectional:
            self.d_model = 2*hidden_size
        else:
            self.d_model = hidden_size
        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")
        self.group_attention = GroupAttention(8,self.d_model,dropout_ratio,in_word2idx)
        self.onelayer = GroupATTEncoder(self.d_model,8,self.d_model,dropout_ratio,d_ff,N,in_word2idx)
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

    def forward(self,input_seq, input_embeddings, input_length, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length,
                                                                          batch_first=True, enforce_sorted=False)
        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)

        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        src_mask = self.group_attention.get_mask(input_seq)
        outputs = self.onelayer(outputs,src_mask)
        return outputs, hidden_states


class HMSEncoder(nn.Module):
    def __init__(self, embed_model, hidden_size=512, span_size=0, dropout=0.4):
        super(HMSEncoder, self).__init__()
        self.hidden_size = hidden_size
        embed_size = embed_model.embedding_dim
        
        self.embedding = embed_model
        # word encoding
        self.word_rnn = nn.GRU(embed_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # span encoding
        # span sequence
        self.span_attn = Attention(self.hidden_size, mix=True, fn=True)
        self.pos_enc = PositionalEncoding(span_size, hidden_size)
        # merge subtree/word node
        self.to_parent = Attention(self.hidden_size, mix=True, fn=True)
        return

    def bi_combine(self, output, hidden):
        # combine forward and backward LSTM
        # (num_layers * num_directions, batch, hidden_size).view(num_layers, num_directions, batch, hidden_size)
        hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden
    
    def dependency_encode(self, word_output, tree):
        word, rel, left, right = tree
        children = left + right
        word_vector = word_output[:, word]
        if len(children) == 0:
            vector = word_vector
        else:
            children_vector = [self.dependency_encode(word_output, child).unsqueeze(1) for child in children]
            children_vector = torch.cat(children_vector, dim=1)
            query = word_vector.unsqueeze(1)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        return vector

    def forward(self, input_var, input_lengths, span_length, tree=None):
        use_cuda = span_length.is_cuda
        pad_hidden = torch.zeros(1, self.hidden_size)
        if use_cuda:
            pad_hidden = pad_hidden.cuda()
        
        word_outputs = []
        span_inputs = []
        
        input_vars = input_var
        trees = tree
        bi_word_hidden = None
        for span_index, input_var in enumerate(input_vars):
            input_length = input_lengths[span_index]

            # word encoding
            embedded = self.embedding(input_var)
            embedded = self.dropout(embedded)
            # at least 1 word in some full padding span
            pad_input_length = input_length.clone()
            pad_input_length[pad_input_length == 0] = 1
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, pad_input_length, batch_first=True, enforce_sorted=False)
            word_output, bi_word_hidden = self.word_rnn(embedded, bi_word_hidden)
            word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)
            word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)
            
            # tree encoding
            span_span_input = []
            for data_index, data_word_output in enumerate(word_output):
                data_word_output = data_word_output.unsqueeze(0)
                tree = trees[span_index][data_index]
                if tree is not None:
                    data_span_input = self.dependency_encode(data_word_output, tree)
                else:
                    data_span_input = pad_hidden
                span_span_input.append(data_span_input)
            span_input = torch.cat(span_span_input, dim=0)
            span_inputs.append(span_input.unsqueeze(1))
            word_outputs.append(word_output)
        
        # span encoding
        span_input = torch.cat(span_inputs, dim=1)
        span_mask = get_mask(span_length, span_input.size(1))
        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, _ = self.span_attn(span_output, span_output, span_mask)
        span_output = span_output * (span_mask == 0).unsqueeze(-1)
        dim0 = torch.arange(span_output.size(0))
        if use_cuda:
            dim0 = dim0.cuda()
        span_hidden = span_output[dim0, span_length - 1].unsqueeze(0)
        return (span_output, word_outputs), span_hidden
