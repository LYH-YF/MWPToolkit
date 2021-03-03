import torch
from torch import nn

class PositionEmbedder(nn.Module):
    '''
    给原始序列添加位置编码
    '''
    def __init__(self, embedding_size,device, dropout=0.1, max_len=1024):
        super(PositionEmbedder, self).__init__()
        
        # 首先初始化为0
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        # sine 和 cosine 来生成位置信息
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.dropout = nn.Dropout(p=dropout)
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
        outputs=self.dropout(outputs)
        return outputs