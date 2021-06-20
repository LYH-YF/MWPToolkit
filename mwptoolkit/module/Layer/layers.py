import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import Attention

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
            encoder_state (FloatTensor): Last cell state of the encoder
                (output of Encoder module).
            context (FloatTensor): Encoded context, with size
                (batch_size, text_len, dim_hidden).

        Return:
            var_embedding (FloatTensor): Embedding of an unknown variable,
                with size (batch_size, dim_context)
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
        # return torch.stack([self.ret] * top2.size(0), 0)

