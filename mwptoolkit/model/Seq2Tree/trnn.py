import torch
from torch import nn

from mwptoolkit.module.Layer.tree_layers import Node,BinaryTree
from mwptoolkit.module.Layer.tree_layers import RecursiveNN
from mwptoolkit.module.Encoder.rnn_encoder import SelfAttentionRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec

class TRNN(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.seq2seq = RNNEncDec(config)
        self.embedder=BaiscEmbedder(config["vocab_size"],config["embedding_size"],config["dropout_ratio"])
        self.attn_encoder=SelfAttentionRNNEncoder(config["embedding_size"],config["hidden_size"],config["num_layers"],\
                                                    config["rnn_cell_type"],config["dropout_ratio"],config["bidirectional"])
        self.recursivenn=RecursiveNN(config["embedding_size"],config["operator_nums"])
    
    def forward(self,seq,seq_length,target=None):
        if target != None:
            self.generate_t()
        else 
    def generate_t(self):
        pass
    def generate_without_t(self):
        pass
    def seq2seq_forward(self,seq,seq_length,target=None):
        return self.seq2seq(seq,seq_length,target)
    def recursivenn_forward(self,seq,seq_length,target=None):
        template=self.seq2seq_forward(seq,seq_length)
    def template2tree(self):
        pass

