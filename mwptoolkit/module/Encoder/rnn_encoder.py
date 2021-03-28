import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import SeqAttention


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