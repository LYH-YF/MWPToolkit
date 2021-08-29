# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:46:05
# @File: bert_embedder.py


import torch
from torch import nn
from transformers import BertModel

class BertEmbedder(nn.Module):
    def __init__(self,input_size,pretrained_model_path):
        super(BertEmbedder,self).__init__()
        self.bert=BertModel.from_pretrained(pretrained_model_path)
        
    def forward(self,input_seq):
        output=self.bert(input_seq)[0]
        return output
    def token_resize(self,input_size):
        self.bert.resize_token_embeddings(input_size)