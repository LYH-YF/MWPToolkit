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

        output = F.tanh(self.linear_out(combined.view(-1, 2*self.hidden_size)))\
                            .view(batch_size, -1, self.hidden_size)

        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn