import torch
from torch import nn


class BaiscEmbedder(nn.Module):
    r'''
    Basic embedding layer
    '''
    def __init__(self,input_size,embedding_size,dropout_ratio,padding_idx=0):
        super(BaiscEmbedder,self).__init__()
        self.embedder=nn.Embedding(input_size,embedding_size,padding_idx=padding_idx)
        self.dropout=nn.Dropout(dropout_ratio)
    
    def forward(self,input_seq):
        '''Implement the embedding process
        Args:
            input_seq (Torch.Tensor): source sequence, shape [batch_size, sequence_length].
        Retruns:
            - Torch.Tensor: embedding output, shape[batch_size, sequence_length, embedding_size]
        '''
        embedding_output=self.embedder(input_seq)
        embedding_output=self.dropout(embedding_output)

        return embedding_output
