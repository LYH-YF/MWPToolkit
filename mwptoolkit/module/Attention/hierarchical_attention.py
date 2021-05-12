import torch
from torch import nn
import torch.nn.functional as F

def get_mask(encode_lengths, pad_length):
    use_cuda = encode_lengths.is_cuda
    batch_size = encode_lengths.size(0)
    index = torch.arange(pad_length)
    if use_cuda:
        index = index.cuda()
    mask = (index.unsqueeze(0).expand(batch_size, -1) >= encode_lengths.unsqueeze(-1)).byte()
    # save one position for full padding span to prevent nan in softmax
    # invalid value in full padding span will be ignored in span level attention
    mask[mask.sum(dim=-1) == pad_length, 0] = 0
    return mask

class Attention(nn.Module):
    def __init__(self, dim, mix=True, fn=False):
        super(Attention, self).__init__()
        self.mix = mix
        self.fn = fn
        if fn:
            self.linear_out = nn.Linear(dim*2, dim)        
        self.w = nn.Linear(dim*2, dim)
        self.score = nn.Linear(dim, 1)
        return

    def forward(self, output, context, mask=None):
        # output/context: batch_size * seq_len * hidden_size
        # mask: batch_size * seq_len
        batch_size, output_size, _ = output.size()
        input_size = context.size(1)
        # batch_size * output_size * input_size * hidden_size
        in_output = output.unsqueeze(2).expand(-1, -1, input_size, -1)
        in_context = context.unsqueeze(1).expand(-1, output_size, -1, -1)
        score_input = torch.cat((in_output, in_context), dim=-1)
        score_input = F.leaky_relu(self.w(score_input))
        score = self.score(score_input).view(batch_size, output_size, input_size)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, output_size, -1)
            score.data.masked_fill_(mask==1, -float('inf'))
        attn = F.softmax(score, dim=-1)

        if self.mix:
            # (b, o, i) * (b, i, dim) -> (b, o, dim)
            attn_output = torch.bmm(attn, context)
        else:
            attn_output = None

        if self.fn:
            combined = torch.cat((attn_output, output), dim=2)
            attn_output = F.leaky_relu(self.linear_out(combined))
        # attn_output: (b, o, dim)
        # attn  : (b, o, i)
        return attn_output, attn

class HierarchicalAttention(nn.Module):
    def __init__(self, dim):
        super(HierarchicalAttention, self).__init__()
        self.span_attn = Attention(dim, mix=False, fn=False)
        self.word_attn = Attention(dim, mix=True, fn=False)
        return

    def forward(self, output, span_context, word_contexts, span_mask=None, word_masks=None):
        batch_size, output_size, _ = output.size()
        _, span_size, hidden_size = span_context.size()
        _, span_attn = self.span_attn(output, span_context, span_mask)
        word_outputs = []
        for word_context, word_mask in zip(word_contexts, word_masks):
            word_output, _ = self.word_attn(output, word_context, word_mask)
            word_outputs.append(word_output.unsqueeze(-2))
        
        # normal
        # batch_size * output_size * span_size * hidden_size
        word_output = torch.cat(word_outputs, dim=-2)
        # (batch_size*output_size) * span_size * hidden_size
        word_output = word_output.view(-1, span_size, hidden_size)
        span_context = span_context.unsqueeze(1).expand(-1, output_size, -1, -1).view(-1, span_size, hidden_size)
        # batch_size * output_size * span_size => (batch_size*output_size) * 1 * span_size
        span_attn = span_attn.view(-1, 1, span_size)
        # (batch_size*output_size) * 1 * hidden_size
        attn_output = torch.bmm(span_attn, (span_context + word_output))
        # batch_size * output_size * hidden_size
        attn_output = attn_output.view(batch_size, output_size, hidden_size)
        return attn_output
