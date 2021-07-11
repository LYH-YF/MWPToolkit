import torch
import math
from torch import nn
from torch.nn import functional as F

from transformers.modeling_bert import gelu_new as gelu_bert

from mwptoolkit.module.Attention.multi_head_attention import EPTMultiHeadAttention
from mwptoolkit.module.Attention.group_attention import GroupAttention
from mwptoolkit.utils.utils import clones


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
    def __init__(self, embedding_size, ffn_size, num_heads, attn_dropout_ratio=0.0, attn_weight_dropout_ratio=0.0, ffn_dropout_ratio=0.0, with_external=False):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = EPTMultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
        self.feed_forward_1 = nn.Linear(embedding_size, ffn_size)
        self.feed_forward_2 = nn.Linear(ffn_size, embedding_size)

        self.attn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.ffn_dropout = nn.Dropout(ffn_dropout_ratio)

        self.with_external = with_external

        if self.with_external:
            self.external_multi_head_attention = EPTMultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
            self.external_layer_norm = nn.LayerNorm(embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.feed_forward_1.weight, std=0.02)
        nn.init.normal_(self.feed_forward_2.weight, std=0.02)
        nn.init.constant_(self.feed_forward_1.bias, 0.)
        nn.init.constant_(self.feed_forward_2.bias, 0.)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, kv=None, self_padding_mask=None, self_attn_mask=None, external_states=None, external_padding_mask=None):
        residual = x
        if kv is None:
            x, self_attn_weights = self.multi_head_attention(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        else:
            x, self_attn_weights = self.multi_head_attention(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        x = self.attn_dropout(x)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn_weights = self.external_multi_head_attention(query=x, key=external_states, value=external_states, key_padding_mask=external_padding_mask)
            x = self.attn_dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn_weights = None

        residual = x
        x = self.feed_forward_2(self.gelu(self.feed_forward_1(x)))
        x = self.ffn_dropout(x)
        x = self.ffn_layer_norm(residual + x)

        return x, self_attn_weights, external_attn_weights


# class Encoder(nn.Module):
#     "Core encoder is a stack of N layers"

#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


# class GAEncoderLayer(nn.Module):
#     "Group attention based encoder layer"

#     def __init__(self, size, h, d_model, dropout_ratio, d_ff, in_word2idx):
#         super(GAEncoderLayer, self).__init__()
#         self.self_attn = GroupAttention(h, d_model, dropout_ratio, in_word2idx)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_ratio)

#         self.sublayer = clones(SublayerConnection(size, dropout_ratio), 2)
#         self.size = size

#     def forward(self, x, mask):
#         "Follow Figure 1 (left) for connections."
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)


# class GAEncoderLayer(nn.Module):
#     "Group attention based encoder layer"

#     def __init__(self, size, h, d_model, dropout_ratio, d_ff):
#         super(GAEncoderLayer, self).__init__()
#         self.self_attn = GroupAttention(h, d_model, dropout_ratio)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_ratio)

#         #self.sublayer = clones(SublayerConnection(size, dropout_ratio), 2)
#         self.attn_layer_norm = nn.LayerNorm(size)
#         self.attn_dropout = nn.Dropout(dropout_ratio)

#         self.ff_layer_norm = nn.LayerNorm(size)
#         self.ff_dropout = nn.Dropout(dropout_ratio)

#         self.size = size

#     def forward(self, x, mask):
#         x = self.attn_layer_norm(x)
#         x = x + self.attn_dropout(self.self_attn(x, x, x, mask))

#         x = self.ff_layer_norm(x)
#         x = x + self.ff_dropout(self.feed_forward(x))

#         return x


class GAEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GAEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        #self.norm = LayerNorm(size)
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."

#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EPTTransformerLayer(nn.Module):
    """
    Class for Transformer Encoder/Decoder layer (follows the paper, 'Attention is all you need')
    """

    def __init__(self, hidden_dim = None, num_decoder_heads = None, layernorm_eps = None,intermediate_dim= None):
        """
        Initialize TransformerLayer class

        :param ModelConfig config: Configuration of this Encoder/Decoder layer
        """
        super().__init__()

        # Self-attention layer
        self.attn = EPTMultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_decoder_heads,
                                       layernorm_eps=layernorm_eps, dropout=0.0)
        # Source-Target attention layer
        self.mem = EPTMultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_decoder_heads,
                                       layernorm_eps=layernorm_eps, dropout=0.0)

        # Dropout for self-attention
        self.dropout_attn = nn.Dropout(0.0)
        # Dropout for source-target attention
        self.dropout_mem = nn.Dropout(0.0)
        # Dropout for expansion before outputting
        self.dropout_expand = nn.Dropout(0.0)
        # Dropout for outputting
        self.dropout_out = nn.Dropout(0.0)

        # Linear transformation layer for expansion (H -> I) where I = vector dimension of intermediate state
        self.lin_expand = nn.Linear(hidden_dim, intermediate_dim)
        # Linear transformation layer for output (I -> H)
        self.lin_collapse = nn.Linear(intermediate_dim, hidden_dim)

        # Post Layer Normalization for self-attention
        self.norm_attn = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        # Post Layer Normalization for source-target attention
        self.norm_mem = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        # Post Layer Normalization for outputting
        self.norm_out = nn.LayerNorm(hidden_dim, eps=layernorm_eps)

    def forward(self, target, target_ignorance_mask=None, target_attention_mask=None,
                memory=None, memory_ignorance_mask=None):
        """
        Forward-computation of Transformer Encoder/Decoder layers

        :param torch.Tensor target:
            FloatTensor indicating Sequence of target vectors. Shape [B, T, H]
            where B = batch size, T = length of target sequence, H = vector dimension of hidden state
        :param torch.Tensor target_ignorance_mask:
            BoolTensor indicating Mask for target tokens that should be ignored. Shape [B, T].
        :param torch.Tensor target_attention_mask:
            BoolTensor indicating Target-to-target Attention mask for target tokens. Shape [T, T].
        :param torch.Tensor memory:
            FloatTensor indicating Sequence of source vectors. Shape [B, S, H]
            where S = length of source sequence
            This can be None when you want to use this layer as an encoder layer.
        :param torch.Tensor memory_ignorance_mask:
            BoolTensor indicating Mask for source tokens that should be ignored. Shape [B, S].
        :rtype: torch.FloatTensor
        :return: Decoder hidden states per each target token, shape [B, S, H].
        """
        # Compute self-attention
        attented = self.attn(query=target, attention_mask=target_attention_mask,
                             key_ignorance_mask=target_ignorance_mask)
        target = target + self.dropout_attn(attented)
        target = self.norm_attn(target)

        # Compute attention over targets with source as queries.
        if memory is not None:
            attented = self.mem(query=target, key_value=memory, key_ignorance_mask=memory_ignorance_mask)
            target = target + self.dropout_mem(attented)
            target = self.norm_mem(target)

        # Pass linear transformations
        output = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(target))))
        target = target + self.dropout_out(output)
        target = self.norm_out(target)

        return target

