import math
import torch
from torch import nn

class PositionEmbedder_x(nn.Module):
    '''
    给原始序列添加位置编码
    '''
    def __init__(self, embedding_size, max_len=1024):
        super(PositionEmbedder_x, self).__init__()
        
        # 首先初始化为0
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        # sine 和 cosine 来生成位置信息
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, input_embedding):
        '''
        Args:
            input_embedding: torch.Tensor, [batch_size, seq_length, embedding_size].
        '''
        # 词经过嵌入层后，再加上位置信息
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