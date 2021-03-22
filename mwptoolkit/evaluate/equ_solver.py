def joint_number_(text_list): #match longer fraction such as ( 1 / 1000000 )
    #text_list='( | ( 5 / 7 ) - ( 1 / 14 ) + ( 5 / 6 ) ) / ( 5 / 42 ) ．'.split(" ")
    #text_list="计 ( 123 (1/10000) ) 算 ( 1 / 2 ) + ( 1 / 6 ) + ( 1 / 12 ) + ( 1 / 20 ) + … + ( 1 / 380 ) = 多 少 ．".split()
    text_list="苹 果 树 比 梨 树 少 ( 3 / 8 ) ， 梨 树 比 苹 果 树 多 ( ( ( ) ) / ( ( ) ) )".split(" ")
    new_list=[]
    i=0
    while i < len(text_list):
        if text_list[i] == '(':
            try:
                j=text_list[i:].index(')')
                if i+1==i+j:
                    j=None
                if "(" in text_list[i+1:i+j+1]:
                    j=None
            except:
                j=None
            if j:
                stack=[]
                flag=True
                idx=0
                for temp_idx,word in enumerate(text_list[i:i+j+1]):
                    if word in ["(",")","/"] or word.isdigit():
                        stack.append(word)
                        idx=temp_idx
                    else:
                        flag=False
                        break
                if "/" not in stack:
                    flag=False
                if flag:
                    number=''.join(stack)
                    new_list.append(number)
                else:
                    for word in stack:
                        new_list.append(word)
                i+=idx+1
            else:
                new_list.append(text_list[i])
                i+=1
        else:
            new_list.append(text_list[i])
            i+=1
    return new_list
#joint_number_([])
from typing import Counter
import torch
from torch import nn, stack
from torch.nn import functional as F
import stanza
import copy
def get_group_num(doc,sent_len,num_pos):
    heads_deprel_upos=doc.get(['head','deprel','upos','text'])
    num_pos=[pos-sent_len for pos in num_pos]
    for idx,x in enumerate(heads_deprel_upos):
        print(idx,x)
    for n_pos in num_pos:
        pos_stack=[]
        group_num=[]
        #group_num.append(n_pos)
        pos_stack.append([n_pos,heads_deprel_upos[n_pos][1]])
        count=0
        head_pos=heads_deprel_upos[n_pos][0]
        head_dep=heads_deprel_upos[n_pos][1]
        for idx,x in enumerate(heads_deprel_upos):
            if heads_deprel_upos[idx][0]==head_pos and n_pos!=idx:
                deprel=heads_deprel_upos[idx][1]
                pos_stack.append([idx,deprel])
        while pos_stack:
            count+=1
            pos_dep=pos_stack.pop(0)
            pos=pos_dep[0]
            dep=pos_dep[1]
            head_pos=heads_deprel_upos[pos][0]-1
            upos=heads_deprel_upos[pos][2]
            if upos not in ['NOUN','NUM','ADJ','VERB','DET', 'SYM']:
                continue
            elif upos == 'NOUN' and dep not in ['compound','nsubj:pass','nsubj','compound']:
                continue
            elif upos == 'VERB' and dep not in ['conj','root']:
                continue
            elif upos == 'ADJ' and dep not in ['amod']:
                continue
            elif upos == 'DET' and dep not in ['advmod']:
                continue
            elif upos == 'SYM' and dep not in ['obl']:
                continue
            else:
                group_num.append(pos)
            if head_pos>=0:
                head_dep=heads_deprel_upos[head_pos][1]
                if [head_pos,head_dep] in pos_stack:
                    pass
                else:
                    pos_stack.append([head_pos,head_dep])
        print(count)
        print(group_num)
    return []
def get_group_num_(token_list,sent_len,num_pos):
    group_nums=[]
    num_pos=[pos-sent_len for pos in num_pos]
    for n_pos in num_pos:
        pos_stack=[]
        group_num=[]
        pos_stack.append([n_pos,token_list[n_pos]["deprel"]])
        count=0
        head_pos=token_list[n_pos]['head']
        for idx,x in enumerate(token_list):
            if x['head']==head_pos and n_pos!=idx:
                deprel=x["deprel"]
                pos_stack.append([idx,deprel])
        while pos_stack:
            count+=1
            pos_dep=pos_stack.pop(0)
            pos=pos_dep[0]
            dep=pos_dep[1]
            head_pos=token_list[pos]['head']-1
            upos=token_list[pos]['upos']
            if upos not in ['NOUN','NUM','ADJ','VERB','DET', 'SYM']:
                continue
            elif upos == 'NOUN' and dep not in ['compound','nsubj:pass','nsubj','compound']:
                continue
            elif upos == 'VERB' and dep not in ['conj','root']:
                continue
            elif upos == 'ADJ' and dep not in ['amod']:
                continue
            elif upos == 'DET' and dep not in ['advmod']:
                continue
            elif upos == 'SYM' and dep not in ['obl']:
                continue
            else:
                group_num.append(pos+sent_len)
            if head_pos>=0:
                head_dep=token_list[head_pos]['deprel']
                if [head_pos,head_dep] in pos_stack:
                    pass
                else:
                    pos_stack.append([head_pos,head_dep])
        group_nums.append(group_num)
    return group_nums
# nlp=stanza.Pipeline('en',processors='depparse,tokenize,pos,lemma',tokenize_pretokenized=True)
# text="348 teddy bears are sold for $ 23 each . There are total 470 teddy bears in a store and the remaining teddy bears are sold for $ 17 each . How much did the store earn after selling all the teddy bears ."
# doc=nlp(text)
# print(doc)
# num_pos=[0, 7,13,28]
# token_lists=doc.to_dict()
# sentences=text.split(".")
# sent_len=[]
# num_pos=[0, 7,13,28]
# num_pos_list=[]
# last_pos_list=[]
# l=0
# for idx,sentence in enumerate(sentences):
#     sentence=sentence+' .'
#     sentences[idx]=sentence
#     sent_len.append(l)
#     l+=len(sentence.split(" "))
#     n_pos=[pos for pos in num_pos if pos <l ]
#     for pos in n_pos:
#         num_pos.remove(pos)
#     num_pos_list.append(n_pos)
# for idx,token_list in enumerate(token_lists):
#     doc=nlp(sentences[idx])
#     #print(doc)
#     token_list=doc.to_dict()[0]
#     group=get_group_num_(token_list,sent_len[idx],num_pos_list[idx])
#     print(group)
# exit()
#text="在 一 正方形 花池 的 4 周 栽 了 44 棵 柳树 ， 每 两棵 柳树 之间 的 间隔 是 20 米 ， 这个 正方形 的 周长 = 多少 米 ？"
#text="在一正方形花池的4周栽了44棵柳树，每两棵柳树之间的间隔是20米，这个正方形的周长=多少米？"
#text=['在','一','正方形','花池','的','4','周','栽','了','44','棵','柳树','，','每','两棵', '柳树', '之间', '的', '间隔', '是', '20', '米', '，', '这个','正方形' ,'的', '周长', '=', '多少' ,'米', '？']
# nlp=stanza.Pipeline('en',processors='depparse,tokenize,pos,lemma',tokenize_pretokenized=True)
# text="348 teddy bears are sold for $ 23 each. There are total 470 teddy bears in a store and the remaining teddy bears are sold for $ 17 each. How much did the store earn after selling all the teddy bears."
# sentences=text.split(".")
# sent_len=[]
# num_pos=[0, 7,13,28]
# num_pos_list=[]
# last_pos_list=[]
# l=0
# for idx,sentence in enumerate(sentences):
#     sentence=sentence+' .'
#     sentences[idx]=sentence
#     sent_len.append(l)
#     l+=len(sentence.split(" "))
#     n_pos=[pos for pos in num_pos if pos <l ]
#     for pos in n_pos:
#         num_pos.remove(pos)
#     num_pos_list.append(n_pos)
# for sent,s_len,sub_n_pos in zip(sentences,sent_len,num_pos_list):
#     doc=nlp(sent+' .')
#     print(doc)
#     g_n=get_group_num(doc,s_len,sub_n_pos)
#doc = nlp("镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 。 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．")
#doc = zh_nlp(text)
#doc.sentences[0].print_dependencies()
#print(doc)
#nlp = stanza.Pipeline('zh-hans') # This sets up a default neural pipeline in English
#text='地图 与 现实 的 比例 是 1:500 。'
#doc = nlp(text)
#doc.sentences[0].print_dependencies()
#doc = nlp("348 teddy bears are sold for $ 23 each. There are total 470 teddy bears in a store and the remaining teddy bears are sold for $ 17 each. How much did the store earn after selling all the teddy bears.")
#print(doc)
# doc.sentences[0].print_dependencies()
# doc.sentences[1].print_dependencies()
# doc.sentences[2].print_dependencies()
print(1)
# class SeqAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(SeqAttention, self).__init__()
#         self.hidden_size=hidden_size

#         self.linear_out = nn.Linear(hidden_size*2, hidden_size)

#     def forward(self, output, encoder_outputs,mask):
#         '''
#         Args:
#             hidden: shape [batch_size, 1, hidden_size].
#             encoder_outputs: shape [batch_size, sequence_length, hidden_size].
#         '''
#         batch_size = output.size(0)
#         seq_length = encoder_outputs.size(1)
        
#         attn = torch.bmm(output, encoder_outputs.transpose(1,2))
#         if mask is not None:
#             attn.data.masked_fill_(mask, -float('inf'))
#         attn = F.softmax(attn.view(-1, seq_length), dim=1).view(batch_size, -1, seq_length)

#         mix = torch.bmm(attn, encoder_outputs)

#         combined = torch.cat((mix, output), dim=2)

#         output = torch.tanh(self.linear_out(combined.view(-1, 2*self.hidden_size)))\
#                             .view(batch_size, -1, self.hidden_size)

#         # output: (b, o, dim)
#         # attn  : (b, o, i)
#         return output, attn
# attn=SeqAttention(128)
# x=torch.rand((4,20,128))
# y=torch.rand((4,1,128))
# output,atten=attn.forward(y,x,None)
# print(output.size())