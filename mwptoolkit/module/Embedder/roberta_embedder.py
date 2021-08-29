# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:47:51
# @File: roberta_embedder.py


import torch
from torch import nn
from transformers import RobertaModel

class RobertaEmbedder(nn.Module):
    def __init__(self,input_size,pretrained_model_path):
        super(RobertaEmbedder,self).__init__()
        #roberta=RobertaModel.from_pretrained(pretrain_model_path)
        self.roberta=RobertaModel.from_pretrained(pretrained_model_path)
        #self.roberta.resize_token_embeddings(input_size)
    
    def forward(self,input_seq,attn_mask):
        output=self.roberta(input_seq,attention_mask = attn_mask)[0]
        return output
    
    def token_resize(self,input_size):
        self.roberta.resize_token_embeddings(input_size)