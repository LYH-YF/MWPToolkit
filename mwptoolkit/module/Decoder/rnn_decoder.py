import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import SeqAttention

class BasicRNNDecoder(nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) decoder.
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 rnn_cell_type,
                 dropout_ratio=0.0):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == "gru":
            self.decoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == "rnn":
            self.decoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in decoder must in ['lstm', 'gru', 'rnn'].")

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
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None):
        r""" Implement the decoding process.

        Args:
            input_embeddings (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        # hidden_states = hidden_states.contiguous()
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states

class AttentionalRNNDecoder(nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder.
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 context_size,
                 num_dec_layers,
                 rnn_cell_type,
                 dropout_ratio=0.0):
        super(AttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_cell_type = rnn_cell_type

        self.attentioner=SeqAttention(hidden_size,hidden_size)
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'gru':
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'rnn':
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention_dense = nn.Linear(hidden_size, hidden_size)

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
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_cell_type))

    def forward(self, input_embeddings, hidden_states=None, encoder_outputs=None, encoder_masks=None):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (Torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (Torch.Tensor): initial hidden states, default: None.
            encoder_outputs (Torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_masks (Torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple:
                - Torch.Tensor: output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                - Torch.Tensor: hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        decode_length = input_embeddings.size(1)

        all_outputs = []
        for step in range(decode_length):
            output, hidden_states = self.decoder(input_embeddings[:,step,:].unsqueeze(1), hidden_states)

            output, attn = self.attentioner(output, encoder_outputs,encoder_masks)

            output=self.attention_dense(output.view(-1,self.hidden_size))

            output=output.view(-1,1,self.hidden_size)

            all_outputs.append(output)
        outputs = torch.cat(all_outputs, dim=1)
        return outputs, hidden_states