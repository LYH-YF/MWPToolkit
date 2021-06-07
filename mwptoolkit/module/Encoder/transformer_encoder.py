import torch
from torch import nn

from mwptoolkit.module.Layer.transformer_layer import TransformerLayer,GAEncoderLayer,LayerNorm
from mwptoolkit.utils.utils import clones

class TransformerEncoder(nn.Module):
    r"""
    The stacked Transformer encoder layers.
    """
    def __init__(self,
                 embedding_size,
                 ffn_size,
                 num_encoder_layers,
                 num_heads,
                 attn_dropout_ratio=0.0,
                 attn_weight_dropout_ratio=0.0,
                 ffn_dropout_ratio=0.0):
        super(TransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.transformer_layers.append(
                TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio,
                                 ffn_dropout_ratio))

    def forward(self, x, kv=None, self_padding_mask=None, output_all_encoded_layers=False):
        r""" Implement the encoding process step by step.

        Args:
            x (Torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            kv (Torch.Tensor): the cached history latent vector, shape: [batch_size, sequence_length, embedding_size], default: None.
            self_padding_mask (Torch.Tensor): padding mask of target sequence, shape: [batch_size, sequence_length], default: None.
            output_all_encoded_layers (Bool): whether to output all the encoder layers, default: ``False``.

        Returns:
            Torch.Tensor: output features, shape: [batch_size, sequence_length, ffn_size].
        """
        all_encoded_layers = []
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask)
            all_encoded_layers.append(x)
        if output_all_encoded_layers:
            return all_encoded_layers
        return all_encoded_layers[-1]


class GroupATTEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(GroupATTEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# class GroupATTEncoder(nn.Module):
#     "Core encoder is a stack of N layers"

#     def __init__(self, 
#                 size,
#                 num_heads,
#                 d_model,
#                 dropout_ratio,
#                 ffn_size,
#                 num_encoder_layers):
#         super(GroupATTEncoder, self).__init__()
#         #self.layers = clones(layer, N)
#         #self.norm = LayerNorm(layer.size)
#         self.layers = nn.ModuleList()
#         for _ in range(num_encoder_layers):
#             self.layers.append(
#                 GAEncoderLayer(size,num_heads,d_model,dropout_ratio,ffn_size)
#             )
#         #self.norm = LayerNorm(size)
#         self.norm = nn.LayerNorm(size)

#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)
