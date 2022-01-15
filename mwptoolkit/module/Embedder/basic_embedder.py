# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:45:57
# @File: basic_embedder.py


import torch
from torch import nn

from mwptoolkit.utils.enum_type import SpecialTokens


class BasicEmbedder(nn.Module):
    """
    Basic embedding layer
    """
    def __init__(self, input_size, embedding_size, dropout_ratio, padding_idx=0):
        super(BasicEmbedder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedder = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_seq):
        r'''Implement the embedding process
        Args:
            input_seq (torch.Tensor): source sequence, shape [batch_size, sequence_length].
        
        Retruns:
            torch.Tensor: embedding output, shape [batch_size, sequence_length, embedding_size].
        '''
        embedding_output = self.embedder(input_seq)
        embedding_output = self.dropout(embedding_output)

        return embedding_output
    
    def init_embedding_params(self, sentences, vocab):
        import numpy as np
        from gensim.models import word2vec
        
        model = word2vec.Word2Vec(sentences, vector_size=self.embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        # sos_idx = vocab.index(SpecialTokens.SOS_TOKEN)
        # sos_idx = vocab.index(SpecialTokens.EOS_TOKEN)
        for idx in range(len(vocab)):
            if idx != pad_idx:
                try:
                    emb_vectors.append(np.array(model.wv[vocab[idx]]))
                except:
                    emb_vectors.append(np.random.randn((self.embedding_size)))
            else:
                emb_vectors.append(np.zeros((self.embedding_size)))
        emb_vectors = np.array(emb_vectors)
        self.embedder.weight.data.copy_(torch.from_numpy(emb_vectors))

