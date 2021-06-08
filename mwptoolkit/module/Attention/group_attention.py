import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mwptoolkit.utils.utils import clones

# def group_mask(batch,type="self",pad=0):
#     length = batch.shape[1]
#     lis = []
#     if type=="self":
#         for tok in batch:
#             mask = torch.zeros(tok.shape)
#             mask = torch.unsqueeze(mask,-1)
#             for ele in tok:
#                 if ele == pad:
#                     copy = torch.zeros(length)
#                 else:
#                     copy = torch.clone(tok)
#                     if ele != 1000:copy[copy == 1000] = 0
#                     copy[copy != ele] = 0
#                     copy[copy == ele] = 1
#                     #print("self copy",copy)
#                 copy = torch.unsqueeze(copy,-1)
#                 mask = torch.cat([mask,copy.float()],dim=1)
#             mask = mask[:,1:]
#             mask = mask.transpose(0,1)
#             #mask = np.expand_dims(mask,0)
#             mask = torch.unsqueeze(mask,0)
#             lis.append(mask)
#         #res = np.concatenate(tuple(lis))
#         res = torch.cat(lis)
#     elif type=="between":
#         for tok in batch:
#             # mask = np.zeros(tok.shape)
#             # mask = np.expand_dims(mask,-1)
#             mask = torch.zeros(tok.shape)
#             mask = torch.unsqueeze(mask,-1)
#             for ele in tok:
#                 if ele == pad:
#                     copy = torch.zeros(length)
#                     #copy = np.zeros(length)
#                 else:
#                     copy = torch.clone(tok)
#                     copy[copy==1000] = 0
#                     copy[copy ==ele] = 0
#                     copy[copy!= 0] = 1
#                     '''
#                     copy[copy != ele and copy != 1000] = 1
#                     copy[copy == ele or copy == 1000] = 0
#                     '''
#                 # copy = np.expand_dims(copy,-1)
#                 # mask = np.concatenate((mask,copy),axis=1)
#                 copy = torch.unsqueeze(copy,-1)
#                 mask = torch.cat([mask,copy.float()],dim=1)
#             mask = mask[:,1:]
#             mask = mask.transpose(0,1)
#             #mask = np.expand_dims(mask,0)
#             mask = torch.unsqueeze(mask,0)
#             lis.append(mask)
#         #res = np.concatenate(tuple(lis))
#         res = torch.cat(lis)
#     elif type == "question":
#         for tok in batch:
#             # mask = np.zeros(tok.shape)
#             # mask = np.expand_dims(mask,-1)
#             mask = torch.zeros(tok.shape)
#             mask = torch.unsqueeze(mask,-1)
#             for ele in tok:
#                 if ele == pad:
#                     #copy = np.zeros(length)
#                     copy = torch.zeros(length)
#                 else:
#                     copy = torch.clone(tok)
#                     copy[copy != 1000] = 0
#                     copy[copy == 1000] = 1
#                 if ele==1000:
#                     copy[copy==0] = -1
#                     copy[copy==1] = 0
#                     copy[copy==-1] = 1
#                 # copy = np.expand_dims(copy,-1)
#                 # mask = np.concatenate((mask,copy),axis=1)
#                 copy = torch.unsqueeze(copy,-1)
#                 mask = torch.cat([mask,copy.float()],dim=1)
#             mask = mask[:,1:]
#             mask = mask.transpose(0,1)
#             #mask = np.expand_dims(mask,0)
#             mask = torch.unsqueeze(mask,0)
#             lis.append(mask)
#         #res = np.concatenate(tuple(lis))
#         res = torch.cat(lis)
#     else:
#         return "error"
#     return res

def group_mask(batch,type="self",pad=0):
    length = batch.shape[1]
    lis = []
    if type=="self":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    #print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type=="between":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy==1000] = 0
                    copy[copy ==ele] = 0
                    copy[copy!= 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:
                	copy[copy==0] = -1
                	copy[copy==1] = 0
                	copy[copy==-1] = 1
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:return "error"
    return res

def src_to_mask(src, vocab_dict):
    src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idx in src:

        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            if (encode_sen_idx[num] == vocab_dict["．"] or encode_sen_idx[num] == vocab_dict["，"]) \
                    and num != len(encode_sen_idx) - 1:
                token += 1
            if encode_sen_idx[num]==0: mask[num] = 0
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
    return np.array(batch_data_mask_tok)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class GroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        #self.split_list=split_list

    def get_mask(self, src, split_list, pad=0):
        device = src.device
        mask = self.src_to_mask(src, split_list)
        self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
        self.final = torch.cat((self.src_mask_between.to(device).bool(),self.src_mask_self.to(device).bool(),self.src_mask_global.to(device),self.src_mask_question.to(device).bool()),1)
        return self.final.to(device)

    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)
        "Implements Figure 2"

        if mask is not None and len(mask.shape)<4:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # which is linears(query, key, value)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    def src_to_mask(self, src, split_list):
        src = src.cpu().numpy()
        batch_data_mask_tok = []
        for encode_sen_idx in src:

            token = 1
            mask = [0] * len(encode_sen_idx)
            for num in range(len(encode_sen_idx)):
                mask[num] = token
                if encode_sen_idx[num] in split_list and num != len(encode_sen_idx) - 1:
                    token += 1
                if encode_sen_idx[num]==0: mask[num] = 0
            for num in range(len(encode_sen_idx)):
                if mask[num] == token and token != 1:
                    mask[num] = 1000
            batch_data_mask_tok.append(mask)
        return np.array(batch_data_mask_tok)



# class GroupAttention(nn.Module):
#     def __init__(self, h, d_model, dropout):
#         "Take in model size and number of heads."
#         super(GroupAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 3)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#         #self.in_word2idx=in_word2idx
#         # self.ignore_list=[]
#         # # chinese dataset
#         # try:
#         #     self.ignore_list.append(in_word2idx['．'])
#         # except:
#         #     pass
#         # try:
#         #     self.ignore_list.append(self.in_word2idx["，"])
#         # except:
#         #     pass
#         # # english dataset
#         # try:
#         #     self.ignore_list.append(self.in_word2idx["."])
#         # except:
#         #     pass
#         # try:
#         #     self.ignore_list.append(self.in_word2idx[","])
#         # except:
#         #     pass

#     # def get_mask(self,src,pad=0):
#     #     device=src.device
#     #     mask = self.src_to_mask(src)
#     #     self.src_mask_self = group_mask(mask,"self",pad).bool().unsqueeze(1)
#     #     self.src_mask_between = group_mask(mask,"between",pad).bool().unsqueeze(1)
#     #     self.src_mask_question = group_mask(mask, "question", pad).bool().unsqueeze(1)
#     #     #self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).int()).unsqueeze(1)
#     #     #self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).int()).unsqueeze(1)
#     #     #self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).int()).unsqueeze(1)
#     #     self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
#     #     self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
#     #     self.final = torch.cat((self.src_mask_between.to(device),self.src_mask_self.to(device),self.src_mask_global.to(device),self.src_mask_question.to(device)),1)
#     #     return self.final.to(device)

#     def forward(self, query, key, value, mask=None):
#         #print("query",query,"\nkey",key,"\nvalue",value)
#         "Implements Figure 2"

#         if mask is not None and len(mask.shape)<4:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         else:
#             mask = torch.cat((mask, mask), 1)
#         nbatches = query.size(0)

#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
#         # which is linears(query, key, value)


#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = self.attention(query, key, value, mask=mask,
#                                  dropout=self.dropout)


#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)
#     # def src_to_mask(self,src):
#     #     src = src.cpu().numpy()
#     #     batch_data_mask_tok = []
#     #     for encode_sen_idx in src:

#     #         token = 1
#     #         mask = [0] * len(encode_sen_idx)
#     #         for num in range(len(encode_sen_idx)):
#     #             mask[num] = token
#     #             # if (encode_sen_idx[num] == self.in_word2idx['．'] or encode_sen_idx[num] == self.in_word2idx["，"]) \
#     #             #         and num != len(encode_sen_idx) - 1:
#     #             if (encode_sen_idx[num] in self.ignore_list) and num != len(encode_sen_idx) - 1:
#     #                 token += 1
#     #             if encode_sen_idx[num]==0:mask[num] = 0
#     #         for num in range(len(encode_sen_idx)):
#     #             if mask[num] == token and token != 1:
#     #                 mask[num] = 1000
#     #         batch_data_mask_tok.append(mask)
#     #     return torch.tensor(batch_data_mask_tok)
#     def attention(self,query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"

#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         p_attn = F.softmax(scores, dim=-1)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         return torch.matmul(p_attn, value), p_attn