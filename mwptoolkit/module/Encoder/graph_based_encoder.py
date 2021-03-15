import torch
from torch import nn

from mwptoolkit.module.Graph.graph_module import Graph_Module

class GraphBasedEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,rnn_cell_type,bidirectional, num_layers=2, dropout_ratio=0.5):
        super(GraphBasedEncoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

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
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_embedding, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_embedding, input_lengths,batch_first=True, enforce_sorted=False)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.encoder(packed, pade_hidden)
        pade_outputs, hidden_states = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs,batch_first=True)

        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        #pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output
