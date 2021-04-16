from torch import nn
import torch
from mwptoolkit.utils.utils import clones
def src_to_mask(src):
    src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idx in src:

        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            if (encode_sen_idx[num] == data_loader.vocab_dict["．"] or encode_sen_idx[num] == data_loader.vocab_dict["，"]) \
                    and num != len(encode_sen_idx) - 1:
                token += 1
            if encode_sen_idx[num]==0:mask[num] = 0
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
    return torch.tensor(batch_data_mask_tok)
def group_mask(batch,type="self",pad=0):
    length = batch.shape[1]
    lis = []
    if type=="self":
        for tok in batch:
            #mask = np.zeros(tok.shape)
            #mask = np.expand_dims(mask,-1)
            mask = torch.zeros(tok.shape)
            mask = torch.unsqueeze(mask,-1)
            for ele in tok:
                if ele == pad:
                    #copy = np.zeros(length)
                    copy = torch.zeros(length)
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
                #copy = np.expand_dims(copy,-1)
                #mask = np.concatenate((mask,copy),axis=1)
                copy = torch.unsqueeze(copy,-1)
                mask = torch.cat([mask,copy],dim=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            #mask = np.expand_dims(mask,0)
            mask = torch.unsqueeze(mask,0)
            lis.append(mask)
        #res = np.concatenate(tuple(lis))
        res = torch.cat(lis)
    elif type=="between":
        for tok in batch:
            # mask = np.zeros(tok.shape)
            # mask = np.expand_dims(mask,-1)
            mask = torch.zeros(tok.shape)
            mask = torch.unsqueeze(mask,-1)
            for ele in tok:
                if ele == pad:
                    copy = torch.zeros(length)
                    #copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy==1000] = 0
                    copy[copy ==ele] = 0
                    copy[copy!= 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                # copy = np.expand_dims(copy,-1)
                # mask = np.concatenate((mask,copy),axis=1)
                copy = torch.unsqueeze(copy,-1)
                mask = torch.cat([mask,copy],dim=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            #mask = np.expand_dims(mask,0)
            mask = torch.unsqueeze(mask,0)
            lis.append(mask)
        #res = np.concatenate(tuple(lis))
        res = torch.cat(lis)
    elif type == "question":
        for tok in batch:
            # mask = np.zeros(tok.shape)
            # mask = np.expand_dims(mask,-1)
            mask = torch.zeros(tok.shape)
            mask = torch.unsqueeze(mask,-1)
            for ele in tok:
                if ele == pad:
                    #copy = np.zeros(length)
                    copy = torch.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:
                	copy[copy==0] = -1
                	copy[copy==1] = 0
                	copy[copy==-1] = 1
                # copy = np.expand_dims(copy,-1)
                # mask = np.concatenate((mask,copy),axis=1)
                copy = torch.unsqueeze(copy,-1)
                mask = torch.cat([mask,copy],dim=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            #mask = np.expand_dims(mask,0)
            mask = torch.unsqueeze(mask,0)
            lis.append(mask)
        #res = np.concatenate(tuple(lis))
        res = torch.cat(lis)
    else:
        return "error"
    return res

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

    def get_mask(self,src,pad=0):
        mask = src_to_mask(src)
        self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
        self.final = torch.cat((self.src_mask_between.cuda(),self.src_mask_self.cuda(),self.src_mask_global.cuda(),self.src_mask_question.cuda()),1)
        return self.final.cuda()

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