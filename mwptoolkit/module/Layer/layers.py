# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:04:38
# @File: layers.py


import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import Attention
from mwptoolkit.module.Attention.tree_attention import TreeAttention as Attn

class GenVar(nn.Module):
    """ Module to generate variable embedding.

    Args:
        dim_encoder_state (int): Dimension of the last cell state of encoder
            RNN (output of Encoder module).
        dim_context (int): Dimension of RNN in GenVar module.
        dim_attn_hidden (int): Dimension of hidden layer in attention.
        dim_mlp_hiddens (int): Dimension of hidden layers in the MLP
            that transform encoder state to query of attention.
        dropout_rate (int): Dropout rate for attention and MLP.
    """
    def __init__(self, dim_encoder_state, dim_context,
                 dim_attn_hidden=256, dropout_rate=0.5):
        super(GenVar, self).__init__()
        self.attention = Attention(
            dim_context, dim_encoder_state,
            dim_attn_hidden, dropout_rate)

    def forward(self, encoder_state, context, context_lens):
        """ Generate embedding for an unknown variable.

        Args:
            encoder_state (torch.FloatTensor): Last cell state of the encoder (output of Encoder module).
            context (torch.FloatTensor): Encoded context, with size [batch_size, text_len, dim_hidden].

        Return:
            torch.FloatTensor: Embedding of an unknown variable, with size [batch_size, dim_context]
        """
        attn = self.attention(context, encoder_state.squeeze(0), context_lens)
        return attn


class Transformer(nn.Module):
    def __init__(self, dim_hidden):
        super(Transformer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh()
        )
        self.ret = nn.Parameter(torch.zeros(dim_hidden))
        nn.init.normal_(self.ret.data)

    def forward(self, top2):
        return self.mlp(top2)


class TreeAttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(TreeAttnDecoderRNN, self).__init__()

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
        self.attn = Attn(hidden_size,hidden_size)

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
