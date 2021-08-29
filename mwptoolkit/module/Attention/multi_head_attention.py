# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 11:10:20
# @File: multi_head_attention.py


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mwptoolkit.utils.enum_type import EPT

class MultiHeadAttention(nn.Module):
    r"""Multi-head Attention is proposed in the following paper:
            Attention Is All You Need.
    """
    def __init__(self, embedding_size, num_heads, dropout_ratio=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        assert self.head_size * num_heads == self.embedding_size, "embedding size must be divisible by num_heads"

        self.scaling = self.head_size ** -0.5  # d_k ** -0.5

        self.linear_query = nn.Linear(embedding_size, embedding_size)
        self.linear_key = nn.Linear(embedding_size, embedding_size)
        self.linear_value = nn.Linear(embedding_size, embedding_size)

        nn.init.normal_(self.linear_query.weight, mean=0, std=0.02)
        nn.init.normal_(self.linear_key.weight, mean=0, std=0.02)
        nn.init.normal_(self.linear_value.weight, mean=0, std=0.02)

        self.linear_out = nn.Linear(embedding_size, embedding_size)
        nn.init.normal_(self.linear_out.weight, mean=0, std=0.02)

        self.weight_dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        r"""
        Multi-head attention

        Args:
            query (torch.Tensor): shape [batch_size, tgt_len, embedding_size].
            key (torch.Tensor): shape [batch_size, src_len, embedding_size].
            value (torch.Tensor): shape [batch_size, src_len, embedding_size].
            key_padding_mask (torch.Tensor): shape [batch_size, src_len].
            attn_mask (torch.BoolTensor): shape [batch_size, tgt_len, src_len].

        Return:
            tuple(torch.Tensor, torch.Tensor):
                attn_repre, shape [batch_size, tgt_len, embedding_size].
                attn_weights, shape [batch_size, tgt_len, src_len].
        """
        device=query.device
        batch_size, tgt_len, embedding_size = query.size()
        src_len = key.size(1)
        assert key.size() == value.size()

        q = self.linear_query(query) * self.scaling
        k = self.linear_key(key)
        v = self.linear_value(value)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 3, 1)
        v = v.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k)
        assert list(attn_weights.size()) == [batch_size, self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0).unsqueeze(1).to(device),
                float("-inf")
            )

        if key_padding_mask is not None:
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(device),
                float("-inf")
            )

        attn_weights = self.weight_dropout(F.softmax(attn_weights, dim=-1))
        attn_repre = torch.matmul(attn_weights, v)

        assert list(attn_repre.size()) == [batch_size, self.num_heads, tgt_len, self.head_size]

        attn_repre = attn_repre.transpose(1, 2).contiguous().view(batch_size, tgt_len, embedding_size)
        attn_repre = self.linear_out(attn_repre)

        # maximum attention weight over heads
        attn_weights, _ = attn_weights.max(dim=1)

        return attn_repre, attn_weights

class EPTMultiHeadAttentionWeights(nn.Module):
    """
    Class for computing multi-head attention weights (follows the paper, 'Attention is all you need')

    This class computes dot-product between query Q and key K, i.e.
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttentionWeights class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default.
        :keyword int num_heads: Number of attention heads (N). 12 by default.
        """
        super().__init__()
        self.config = config

        # Check whether D is divisible by H.
        assert self.hidden_dim % self.num_heads == 0, \
            "Hidden dimension %s is not divisible by the number of heads %s." % (self.hidden_dim, self.num_heads)

        # Linear transform for query Q
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Linear transform for key K
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Vector dimension D of input of a single attention head
        self.dim_head = self.hidden_dim // self.num_heads
        # Square root of vector dimension, i.e. \\sqrt{D}
        self.sqrt_dim = self.dim_head ** 0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, head_at_last: bool = True) -> torch.Tensor:
        """
        Compute multi-head attention weights
        
        Args:
            query (torch.Tensor): FloatTensor representing the query matrix with shape [batch_size, query_sequence_length, hidden_size].
            key (torch.Tensor): FloatTensor representing the key matrix with shape [batch_size, key_sequence_length, hidden_size] or [1, key_sequence_length, hidden_size]. By default, this is `None` (Use query matrix as a key matrix)
            key_ignorance_mask (torch.Tensor): BoolTensor representing the mask for ignoring column vector in key matrix, with shape [batch_size, key_sequence_length]. 
                If an element at (b, t) is `True,` then all return elements at batch_size=b, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            attention_mask (torch.Tensor): BoolTensor representing Attention mask for ignoring a key for each query item, with shape [query_sequence_length, key_sequence_length].
                If an element at (s, t) is `True,` then all return elements at sequence_length=s, T=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            head_at_last (bool): Use `True` to make shape of return value be [batch_size, query_sequence_length, key_sequence_length, head_nums].
                If `False,` this method will return [batch_size, head_nums, sequence_length, key_sequence_length]. By default, this is `True`
        
        Returns:
            torch.FloatTensor: FloatTensor of Multi-head Attention weights.
        """

        # If key is None, reuse query matrix Q.
        if key is None:
            key = query

        # Check size & type conditions
        assert query.shape[0] == key.shape[0] or key.shape[0] == 1 or query.shape[0] == 1
        assert key_ignorance_mask is None or (key.shape[:2] == key_ignorance_mask.shape and
                                              key_ignorance_mask.dtype == torch.bool)
        assert attention_mask is None or (query.shape[1] == attention_mask.shape[0] and
                                          key.shape[1] == attention_mask.shape[1] and
                                          attention_mask.dtype == torch.bool)

        # Store length information
        query_len = query.shape[1]
        key_len = key.shape[1]
        batch_size = max(key.shape[0], query.shape[0])

        # Project query & key with linear transformations
        query = self.linear_q(query)
        key = self.linear_k(key)

        # Scale query with sqrt(dim)
        query = query / self.sqrt_dim

        # If key / value has shape [1, T, H], expand it.
        if query.shape[0] == 1:
            query = query.expand(batch_size, -1, -1)
        if key.shape[0] == 1:
            key = key.expand(batch_size, -1, -1)

        # Transform query [B, S, N, H/N] -> [B, N, S, H/N] -> [BN, S, H/N].
        query = query.view(batch_size, query_len, self.num_heads, self.dim_head) \
            .transpose(1, 2).flatten(0, 1).contiguous()
        # Transform key [B, T, N, H/N] -> [B, N, H/N, T] -> [BN, H/T, T].
        key = key.view(batch_size, key_len, self.num_heads, self.dim_head) \
            .permute(0, 2, 3, 1).flatten(0, 1).contiguous()

        # Compute attention weights: [BN, S, T] -> [B, N, S, T]
        attention_weights = torch.bmm(query, key).view(batch_size, self.num_heads, query_len, key_len).contiguous()

        # Apply masks (IMPORTANT!!! This should be applied after GELU for output weights)
        if attention_mask is not None:
            # Recap: attention mask has shape [S, T], which can be broadcasted as [1, 1, S, T].
            attention_weights.masked_fill_(attention_mask, EPT.NEG_INF)

        if key_ignorance_mask is not None:
            # Recap: ignorance mask has shape [B, T] -> [B, 1, 1, T] and apply it.
            attention_weights.masked_fill_(key_ignorance_mask.unsqueeze(1).unsqueeze(1), EPT.NEG_INF)

        if head_at_last:
            # Output will be [B, N, S, T] -> [B, S, T, N]
            return attention_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attention_weights

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of hidden states (H)
        """
        return self.config.get('hidden_dim', 768)

    @property
    def num_heads(self) -> int:
        """
        :rtype: int
        :return: Number of attention heads (N)
        """
        return self.config.get('num_heads', 12)

class EPTMultiHeadAttention(nn.Module):
    """
    Class for computing multi-head attention (follows the paper, 'Attention is all you need')

    This class computes attention over K-V pairs with query Q, i.e.

    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttention class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default
        :keyword int num_heads: Number of attention heads (N). 12 by default
        :keyword float dropout_p: Probability of dropout. 0 by default
        """
        super().__init__()
        # Multi-head Attention Weight layer
        self.attn = EPTMultiHeadAttentionWeights(**config)
        # Dropout over attention weights (as in 'Attention is all you need')
        self.dropout_p=0.0
        self.dropout_attn = nn.Dropout(self.dropout_p)
        # Linear transformations for value and output matrix.
        self.linear_v = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)
        self.linear_out = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)


    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, return_weights: bool = False, **kwargs):
        """
        Compute multi-head attention

        Args:
            query (torch.Tensor): FloatTensor representing the query matrix with shape [batch_size, query_sequence_length, hidden_size].
            key_value (torch.Tensor): FloatTensor representing the key matrix or value matrix with shape [batch_size, key_sequence_length, hidden_size] or [1, key_sequence_length, hidden_size].
                By default, this is `None` (Use query matrix as a key matrix).
            key_ignorance_mask (torch.Tensor): BoolTensor representing the mask for ignoring column vector in key matrix, with shape [batch_size, key_sequence_length].
                If an element at (b, t) is `True,` then all return elements at batch_size=b, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            attention_mask (torch.Tensor): BoolTensor representing Attention mask for ignoring a key for each query item, with shape [query_sequence_length, key_sequence_length].
                If an element at (s, t) is `True,` then all return elements at query_sequence_length=s, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            return_weights (bool): Use `True` to return attention weights. By default, this is `True.`
        
        Returns:
            Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
                If head_at_last is True, return (Attention Output, Attention Weights). Otherwise, return only the Attention Output.
                Attention Output: Shape [batch_size, query_sequence_length, hidden_size].
                Attention Weights: Shape [batch_size, query_sequence_length, key_sequence_length, head_nums].
        """
        # If key_value is None, reuse query matrix Q.
        if key_value is None:
            key_value = query

        # Compute attention scores: [B, N, S, T].
        attn_weights = self.attn(query=query, key=key_value, key_ignorance_mask=key_ignorance_mask,
                                 attention_mask=attention_mask, head_at_last=False)

        # Retrive shape
        batch_size, _, query_len, key_len = attn_weights.shape

        # Compute Softmax values. Shape [B, N, S, T] -> [BN, S, T].
        # For numerical stability, replace NaN with -Inf. (NaN occurs when we should ignore all weights.)
        attn = attn_weights.softmax(dim=-1)
        attn = self.dropout_attn(attn)  # Dropout was applied after softmax in the original paper.
        attn = attn.masked_fill(torch.isnan(attn), 0.0).view(-1, query_len, key_len)

        # Pass linear and transpose value matrix: [1 or B, T, N, H/N] -> [1 or B, N, T, H/N].
        value_size = key_value.shape[0]
        value = self.linear_v(key_value) \
            .view(value_size, key_len, self.attn.num_heads, self.attn.dim_head).transpose(1, 2)

        # If value has shape [1, *], expand it.
        if value_size == 1:
            value = value.expand(batch_size, -1, -1, -1)

        # Flatten dim #0 and #1: [B, N, T, H/N] -> [BN, T, H/N].
        value = value.flatten(0, 1).contiguous()

        # Compute output of weighted sum: [BN, S, H/N] -> [B, N, S, H/N] -> [B, S, N, H/N] -> [B, S, H].
        output = torch.bmm(attn, value) \
            .view(batch_size, self.attn.num_heads, query_len, self.attn.dim_head) \
            .transpose(1, 2).flatten(2, 3).contiguous()

        # Map outputs and return. [B, S, H].
        output = self.linear_out(output)

        if return_weights:
            return output, attn_weights.permute(0, 2, 3, 1).contiguous()
        else:
            # Map outputs and return. [B, S, H].
            return output

