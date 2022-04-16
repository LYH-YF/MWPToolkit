# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:47:08
# @File: position_embedder.py


import math
import torch
from torch import nn

class PositionEmbedder_x(nn.Module):
    def __init__(self, embedding_size, max_len=1024):
        super(PositionEmbedder_x, self).__init__()
        
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, input_embedding):
        '''
        Args:
            input_embedding (torch.Tensor): shape [batch_size, sequence_length, embedding_size].
        '''
        seq_len=input_embedding.size(1)
        #outputs=input_embedding+self.weight[:batch_size,:]
        outputs=input_embedding+self.pe.squeeze()[:seq_len]
        #outputs=self.dropout(outputs)
        return outputs

class PositionEmbedder(nn.Module):
    r"""This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(self, embedding_size, max_length=512):
        super(PositionEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.weights = self.get_embedding(
            max_length,
            embedding_size
        )

    def get_embedding(self,max_length, embedding_size):
        r"""Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
        if embedding_size % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
        return emb

    def forward(self, input_seq, offset=0):
        """
        Args:
            input_seq (torch.Tensor): input sequence, shape [batch_size, sequence_length].
        
        Returns:
            torch.Tensor: position embedding, shape [batch_size, sequence_length, embedding_size].
        """
        batch_size, seq_len = input_seq.size()
        max_position = seq_len + offset
        if self.weights is None or max_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = self.get_embedding(
                max_position,
                self.embedding_size,
            )

        positions = offset + torch.arange(seq_len)
        pos_embeddings = self.weights.index_select(0, positions).unsqueeze(0).expand(batch_size, -1, -1).detach()
        return pos_embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, pos_size, dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.rand(pos_size, dim)
        # (0, 1) => (-1, 1)
        pe = pe * 2 - 1
        self.pe = nn.Parameter(pe)
    
    def forward(self, input):
        output = input + self.pe[:input.size(1)]
        return output
    

class EPTPositionalEncoding(nn.Module):
    """
    Positional encoding that extends trigonometric embedding proposed in 'Attention is all you need'
    """

    def __init__(self, embedding_dim):
        """
        Instantiate positional encoding instance.

        :param int embedding_dim:
            Dimension of embedding vector
        """

        super().__init__()
        #: Dimension of embedding vector
        self.embedding_dim = embedding_dim

        # The output will be c_p * cos(a_p * t + b_p) + d_p * sin(a_p * t + b_p), where t=index and p = 1...embed_dim
        # From "Attention is all you need" paper.
        # Here, b_p = 0 and a_2p = a_{2p+1} = 1 / 10000^{2p/embed_dim}.
        # Thus, we need to define a_p only.
        div_term = (torch.arange(0, embedding_dim) // 2) * 2
        div_term = torch.exp(div_term.float() * (-math.log(10000.0) / embedding_dim))
        # Note: c_p = 1 if p is odd, 0 otherwise and d_p = 1 if p is even, 0 otherwise
        multiplier = torch.zeros(2, embedding_dim, dtype=torch.float)
        multiplier[0, 1::2] = 1.0  # Only use cosine for odd indices
        multiplier[1, 0::2] = 1.0  # Only use sine for even indices

        # Fix a_p, c_p, d_p values.
        self.register_buffer('_div_term', div_term)
        self.register_buffer('multiplier', multiplier)

    @property
    def device(self) -> torch.device:
        """
        Get the device where weights are currently put.
        :rtype: torch.device
        :return: Device instance
        """
        return self._div_term.device

    def before_trigonometric(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute a_p * t + b_p for each index t.
        :param torch.Tensor indices: A Long tensor to compute indices.
        :rtype: torch.Tensor
        :return: Tensor whose values are a_p * t + b_p for each (t, p) entry.
        """
        indices = indices.float()

        # Compute a_p * t.
        return indices * self._div_term

    def forward(self, index_or_range, ignored_index=-1) -> torch.Tensor:
        """
        Compute positional encoding. If this encoding is not learnable, the result cannot have any gradient vector.

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        # we don't need to compute gradients.
        with torch.no_grad():
            return self._forward(index_or_range, ignored_index)

    def _forward(self, index_or_range, ignored_index=-1) -> torch.Tensor:
        """
        Compute positional encoding

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        if type(index_or_range) is int:
            # Build Long Tensor of [0, ..., index-1]
            indices = torch.arange(0, index_or_range)
        elif type(index_or_range) is range:
            # Build Long Tensor of [range]
            indices = torch.as_tensor(list(index_or_range))
        else:
            indices = index_or_range

        # Unsqueeze the last dimension to pass the linear layer.
        indices = indices.unsqueeze(-1)

        # Send indices to device that currently using.
        indices = indices.to(self.device)

        # Now indices will have shape [*, 1], we can apply the linear layer, a_p * t + b_p.
        phase = self.before_trigonometric(indices)

        # Phase has shape [*, E]. Apply cosine and sine function on the phase.
        cos_value = phase.cos()
        sin_value = phase.sin()

        # Retrieve c_p and d_p vectors. These have shape [E].
        cos_multiplier = self.multiplier[0]
        sin_multiplier = self.multiplier[1]

        # To multiply c_p and d_p on [*, E], unsqueeze c_p and d_p to fit [*].
        # Make the dimension of c_p the same
        result_shape = [1] * (phase.dim() - 1) + [-1]
        cos_multiplier = cos_multiplier.view(*result_shape)
        sin_multiplier = sin_multiplier.view(*result_shape)

        # Compute c_p * cos(phase) + d_p * sin(phase). Shape will be [*, E].
        result = cos_value * cos_multiplier + sin_value * sin_multiplier

        # Fill ignored indices as zero.
        ignored_indices = (indices == ignored_index)
        if ignored_indices.any():
            result.masked_fill_(ignored_indices, 0.0)

        # Return value. Shape [*, E]
        return result.contiguous()


class DisPositionalEncoding(nn.Module):

    def __init__(self, embedding_size, max_len):
        super(DisPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.position_encoding = nn.Embedding(max_len, embedding_size)
        self.position_encoding.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, dis_graph, category_num):
        dis_graph_expend = dis_graph.unsqueeze(1)  # B*1*S*S
        ZeroPad = nn.ZeroPad2d(padding=(0, category_num, 0, category_num))  # B*1*S+c*S+C
        dis_graph_expend = ZeroPad(dis_graph_expend)
        input_pos = dis_graph_expend.squeeze(1).long()
        return self.position_encoding(input_pos)

