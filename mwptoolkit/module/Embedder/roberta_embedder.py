import torch
from torch import nn
from transformers import RobertaModel

class RobertaEmbedder(nn.Module):
    def __init__(self,input_size,pretrained_model_path):
        super(RobertaEmbedder,self).__init__()
        #roberta=RobertaModel.from_pretrained(pretrain_model_path)
        self.roberta=RobertaModel.from_pretrained(pretrained_model_path)
        self.roberta.resize_token_embeddings(input_size)
    
    def forward(self,input_seq):
        output=self.roberta(input_seq)[0]
        return output