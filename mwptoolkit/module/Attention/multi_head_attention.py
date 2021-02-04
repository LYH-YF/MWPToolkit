import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    r"""Multi-head Attention is proposed in the following paper:
            Attention Is All You Need.

    Reference:
        https://arxiv.org/abs/1706.03762
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

        self.linear_out = nn.Linear(embedding_size, embedding_size)

        self.weight_dropout = nn.Dropout(dropout_ratio)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.normal_(self.query_proj.weight, std=0.02)
    #     nn.init.normal_(self.key_proj.weight, std=0.02)
    #     nn.init.normal_(self.value_proj.weight, std=0.02)
    #     nn.init.normal_(self.out_proj.weight, std=0.02)
    #     nn.init.constant_(self.query_proj.bias, 0.)
    #     nn.init.constant_(self.key_proj.bias, 0.)
    #     nn.init.constant_(self.value_proj.bias, 0.)
    #     nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        r"""
        Multi-head attention

        Args:
            query: shape: [batch_size, tgt_len, embedding_size]
            key and value: shape: [batch_size, src_len, embedding_size]
            key_padding_mask: shape: [batch_size, src_len]
            attn_mask: shape: [batch_size, tgt_len, src_len]

        Return:
            tuple:
                - attn_repre: shape: [batch_size, tgt_len, embedding_size]
                - attn_weights: shape: [batch_size, tgt_len, src_len]
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
