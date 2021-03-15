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
import torch
from torch import nn
from torch.nn import functional as F
import stanza


text="在 一 正方形 花池 的 4 周 栽 了 44 棵 柳树 ， 每 两棵 柳树 之间 的 间隔 是 20 米 ， 这个 正方形 的 周长 = 多少 米 ？"
#text="在一正方形花池的4周栽了44棵柳树，每两棵柳树之间的间隔是20米，这个正方形的周长=多少米？"

zh_nlp=stanza.Pipeline('zh')
#doc = zh_nlp("镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．")
doc = zh_nlp(text)
[12, 13, 14, 24, 25, 26, 27]
#doc.sentences[0].print_dependencies()
print(doc)
#nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
#doc = nlp("348 teddy bears are sold for $23 each. There are total 470 teddy bears in a store and the remaining teddy bears are sold for $17 each. How much did the store earn after selling all the teddy bears.")
# doc.sentences[0].print_dependencies()
# doc.sentences[1].print_dependencies()
# doc.sentences[2].print_dependencies()

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