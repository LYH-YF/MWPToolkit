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
class SeqAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SeqAttention, self).__init__()
        self.hidden_size=hidden_size

        self.linear_out = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, output, encoder_outputs,mask):
        '''
        Args:
            hidden: shape [batch_size, 1, hidden_size].
            encoder_outputs: shape [batch_size, sequence_length, hidden_size].
        '''
        batch_size = output.size(0)
        seq_length = encoder_outputs.size(1)
        
        attn = torch.bmm(output, encoder_outputs.transpose(1,2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, seq_length), dim=1).view(batch_size, -1, seq_length)

        mix = torch.bmm(attn, encoder_outputs)

        combined = torch.cat((mix, output), dim=2)

        output = torch.tanh(self.linear_out(combined.view(-1, 2*self.hidden_size)))\
                            .view(batch_size, -1, self.hidden_size)

        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn
attn=SeqAttention(128)
x=torch.rand((4,20,128))
y=torch.rand((4,1,128))
output,atten=attn.forward(y,x,None)
print(output.size())