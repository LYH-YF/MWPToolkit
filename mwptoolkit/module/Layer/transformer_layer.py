import torch
import math
from torch import nn

from mwptoolkit.module.Attention.multi_head_attention import MultiHeadAttention

class TransformerLayer(nn.Module):
    r"""Transformer Layer, including
        a multi-head self-attention,
        a external multi-head self-attention layer (only for conditional decoder) and
        a point-wise feed-forward layer.

    Args:
        self_padding_mask (torch.bool): the padding mask for the multi head attention sublayer.
        self_attn_mask (torch.bool): the attention mask for the multi head attention sublayer.
        external_states (torch.Tensor): the external context for decoder, e.g., hidden states from encoder.
        external_padding_mask (torch.bool): the padding mask for the external states.

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """
    def __init__(self, embedding_size, ffn_size, num_heads, attn_dropout_ratio=0.0, attn_weight_dropout_ratio=0.0,
                 ffn_dropout_ratio=0.0, with_external=False):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
        self.feed_forward_1 = nn.Linear(embedding_size, ffn_size)
        self.feed_forward_2 = nn.Linear(ffn_size, embedding_size)

        self.attn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.ffn_dropout = nn.Dropout(ffn_dropout_ratio)

        self.with_external = with_external

        if self.with_external:
            self.external_multi_head_attention = MultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
            self.external_layer_norm = nn.LayerNorm(embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.feed_forward_1.weight, std=0.02)
        nn.init.normal_(self.feed_forward_2.weight, std=0.02)
        nn.init.constant_(self.feed_forward_1.bias, 0.)
        nn.init.constant_(self.feed_forward_2.bias, 0.)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_states=None, external_padding_mask=None):
        residual = x
        if kv is None:
            x, self_attn_weights = self.multi_head_attention(query=x, key=x, value=x,
                                                             key_padding_mask=self_padding_mask,
                                                             attn_mask=self_attn_mask)
        else:
            x, self_attn_weights = self.multi_head_attention(query=x, key=kv, value=kv,
                                                             key_padding_mask=self_padding_mask,
                                                             attn_mask=self_attn_mask)
        x = self.attn_dropout(x)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn_weights = self.external_multi_head_attention(query=x,
                                                                          key=external_states,
                                                                          value=external_states,
                                                                          key_padding_mask=external_padding_mask)
            x = self.attn_dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn_weights = None

        residual = x
        x = self.feed_forward_2(self.gelu(self.feed_forward_1(x)))
        x = self.ffn_dropout(x)
        x = self.ffn_layer_norm(residual + x)

        return x, self_attn_weights, external_attn_weights
