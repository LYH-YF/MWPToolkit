# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 11:11:07
# @File: rnn_decoder.py


import torch
from torch import nn

from mwptoolkit.module.Attention.seq_attention import SeqAttention,Attention,MaskedRelevantScore
from mwptoolkit.module.Layer.layers import Transformer
from mwptoolkit.module.Environment.stack_machine import OPERATIONS

class BasicRNNDecoder(nn.Module):
    r"""
    Basic Recurrent Neural Network (RNN) decoder.
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 rnn_cell_type,
                 dropout_ratio=0.0):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == "gru":
            self.decoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == "rnn":
            self.decoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in decoder must in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None):
        r""" Implement the decoding process.

        Args:
            input_embeddings (torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        # hidden_states = hidden_states.contiguous()
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states

class AttentionalRNNDecoder(nn.Module):
    r"""
    Attention-based Recurrent Neural Network (RNN) decoder.
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 context_size,
                 num_dec_layers,
                 rnn_cell_type,
                 dropout_ratio=0.0):
        super(AttentionalRNNDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_cell_type = rnn_cell_type

        self.attentioner=SeqAttention(hidden_size,hidden_size)
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'gru':
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'rnn':
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")

        self.attention_dense = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_embeddings):
        r""" Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
            hidden_states = (h_0, c_0)
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size).to(device)
        else:
            raise NotImplementedError("No such rnn type {} for initializing decoder states.".format(self.rnn_cell_type))

    def forward(self, input_embeddings, hidden_states=None, encoder_outputs=None, encoder_masks=None):
        r""" Implement the attention-based decoding process.

        Args:
            input_embeddings (torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.
            encoder_outputs (torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_masks (torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)

        decode_length = input_embeddings.size(1)

        all_outputs = []
        for step in range(decode_length):
            output, hidden_states = self.decoder(input_embeddings[:,step,:].unsqueeze(1), hidden_states)

            output, attn = self.attentioner(output, encoder_outputs,encoder_masks)

            output=self.attention_dense(output.view(-1,self.hidden_size))

            output=output.view(-1,1,self.hidden_size)

            all_outputs.append(output)
        outputs = torch.cat(all_outputs, dim=1)
        return outputs, hidden_states


class SalignedDecoder(nn.Module):
    def __init__(self, operations, dim_hidden=300, dropout_rate=0.5, device=None):
        super(SalignedDecoder, self).__init__()
        self.NOOP = operations.NOOP
        self.GEN_VAR = operations.GEN_VAR
        self.ADD = operations.ADD
        self.SUB = operations.SUB
        self.MUL = operations.MUL
        self.DIV = operations.DIV
        self.POWER = operations.POWER
        self.EQL = operations.EQL
        self.N_OPS = operations.N_OPS
        self.PAD = operations.PAD
        self.RAW_EQL = operations.RAW_EQL
        self.BRG = operations.BRG

        self._device = device
        self.transformer_add = Transformer(2 * dim_hidden)
        self.transformer_sub = Transformer(2 * dim_hidden)
        self.transformer_mul = Transformer(2 * dim_hidden)
        self.transformer_div = Transformer(2 * dim_hidden)
        self.transformer_power = Transformer(2 * dim_hidden)
        self.transformers = {
            self.ADD: self.transformer_add,
            self.SUB: self.transformer_sub,
            self.MUL: self.transformer_mul,
            self.DIV: self.transformer_div,
            self.POWER: self.transformer_power,
            self.RAW_EQL: None,
            self.BRG: None}
        self.gen_var = Attention(2 * dim_hidden,
                                 dim_hidden,
                                 dropout_rate=0.0)
        self.attention = Attention(2 * dim_hidden,
                                   dim_hidden,
                                   dropout_rate=dropout_rate)
        self.choose_arg = MaskedRelevantScore(
            dim_hidden * 2,
            dim_hidden * 7,
            dropout_rate=dropout_rate)
        self.arg_gate = torch.nn.Linear(
            dim_hidden * 7,
            3,
            torch.nn.Sigmoid()
        )
        self.rnn = torch.nn.LSTM(2 * dim_hidden,
                                 dim_hidden,
                                 1,
                                 batch_first=True)
        self.op_selector = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, self.N_OPS+1))
        self.op_gate = torch.nn.Linear(
            dim_hidden * 7,
            3,
            torch.nn.Sigmoid()
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer('noop_padding_return',
                             torch.zeros(dim_hidden * 2))
        self.register_buffer('padding_embedding',
                             torch.zeros(dim_hidden * 2))

    def forward(self, context, text_len, operands, stacks,
                prev_op, prev_output, prev_state, number_emb, N_OPS):
        """
        Args:
            context (torch.Tensor): Encoded context, with size [batch_size, text_len, dim_hidden].
            text_len (torch.Tensor): Text length for each problem in the batch.
            operands (list of torch.Tensor): List of operands embeddings for each problem in the batch. Each element in the list is of size [n_operands, dim_hidden].
            stacks (list of StackMachine): List of stack machines used for each problem.
            prev_op (torch.LongTensor): Previous operation, with size [batch, 1].
            prev_arg (torch.LongTensor): Previous argument indices, with size [batch, 1]. Can be None for the first step.
            prev_output (torch.Tensor): Previous decoder RNN outputs, with size [batch, dim_hidden]. Can be None for the first step.
            prev_state (torch.Tensor): Previous decoder RNN state, with size [batch, dim_hidden]. Can be None for the first step.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            op_logits: Logits of operation selection.
            arg_logits: Logits of argument choosing.
            outputs: Outputs of decoder RNN.
            state: Hidden state of decoder RNN.
        """
        batch_size = context.size(0)

        # collect stack states
        stack_states = \
            torch.stack([stack.get_top2().view(-1,) for stack in stacks],
                        dim=0).to(self._device)
        #print('stack_states', stack_states)

        # skip the first step (all NOOP)
        if prev_output is not None:
            # result calculated batch-wise
            batch_result = {
                self.ADD: self.transformer_add(stack_states),
                self.SUB: self.transformer_sub(stack_states),
                self.MUL: self.transformer_mul(stack_states),
                self.DIV: self.transformer_div(stack_states),
                self.POWER: self.transformer_power(stack_states)
            }

        prev_returns = []
        # apply previous op on stacks
        for b in range(batch_size):
            #print('prev_op[b].item()', prev_op[b].item())
            #print(prev_op[b].item(), OPERATIONS.NOOP, OPERATIONS.GEN_VAR, OPERATIONS.EQL); exit()
            # no op
            if prev_op[b].item() == self.NOOP:
                ret = self.noop_padding_return

            elif prev_op[b].item() == self.PAD:
                ret = self.noop_padding_return
            # generate variable
            elif prev_op[b].item() == self.GEN_VAR:
                variable = batch_result[self.GEN_VAR][b]
                operands[b].append(variable)
                stacks[b].add_variable(variable)
                ret = variable
                #print('add_variable', stacks[b]._operands)

            # OPERATIONS.ADD, SUB, MUL, DIV
            elif prev_op[b].item() in [self.ADD, self.SUB,
                                       self.MUL, self.DIV, self.POWER]:
                #print('>>> OPERATIONS.ADD, SUB, MUL, DIV', len(stacks[b]._stack))
                transformed = batch_result[prev_op[b].item()][b]
                #print('transformed', transformed)
                ret = stacks[b].apply_embed_only(
                    prev_op[b].item(),
                    transformed)

            # elif prev_op[b].item() in [self.RAW_EQL, self.BRG]:
            #     ret = stacks[b].apply_embed_only(prev_op[b].item(), None)

            elif prev_op[b].item() == self.EQL:
                ret = stacks[b].apply_eql(prev_op[b].item())

            # push operand
            else:
                #if b == 0: print('>>> push operand', len(stacks[b]._stack))
                stacks[b].push(prev_op[b].item() - N_OPS)
                #ret = operands[b][prev_op[b].item() - N_OPS]
                ret = number_emb[b][prev_op[b].item() - N_OPS]
            prev_returns.append(ret)
            #exit()

        # collect stack states (after applied op)
        stack_states = \
            torch.stack([stack.get_top2().view(-1,) for stack in stacks],
                        dim=0).to(self._device)

        # collect previous returns
        prev_returns = torch.stack(prev_returns)
        prev_returns = self.dropout(prev_returns)

        # decode
        outputs, hidden_state = self.rnn(prev_returns.unsqueeze(1),
                                         prev_state)
        outputs = outputs.squeeze(1)

        # attention
        #print(context, outputs, text_len)
        attention = self.attention(context, outputs, text_len)

        # collect information for op selector
        #print(outputs, stack_states, attention)
        gate_in = torch.cat([outputs, stack_states, attention], -1)
        op_gate_in = self.dropout(gate_in)
        op_gate = self.op_gate(op_gate_in)
        arg_gate_in = self.dropout(gate_in)
        arg_gate = self.arg_gate(arg_gate_in)
        op_in = torch.cat([op_gate[:, 0:1] * outputs,
                           op_gate[:, 1:2] * stack_states,
                           op_gate[:, 2:3] * attention], -1)
        arg_in = torch.cat([arg_gate[:, 0:1] * outputs,
                            arg_gate[:, 1:2] * stack_states,
                            arg_gate[:, 2:3] * attention], -1)
        #print('op_in', op_in.size(), 'arg_in', arg_in.size())
        # op_in = arg_in = torch.cat([outputs, stack_states, attention], -1)

        op_logits = self.op_selector(op_in)

        n_operands, cated_operands = \
            self.pad_and_cat(operands, self.padding_embedding)
        #print('cated_operands, arg_in, n_operands', cated_operands.size(), arg_in.size(), n_operands)
        arg_logits = self.choose_arg(
            cated_operands, arg_in, n_operands)
        #print('arg_logits', arg_logits.size())

        return op_logits, arg_logits, outputs, hidden_state

    def pad_and_cat(self, tensors, padding):
        """ Pad lists to have same number of elements, and concatenate
        those elements to a 3d tensor.

        Args:
            tensors (list of list of Tensors): Each list contains
                list of operand embeddings. Each operand embedding is of
                size (dim_element,).
            padding (Tensor):
                Element used to pad lists, with size (dim_element,).

        Return:
            n_tensors (list of int): Length of lists in tensors.
            tensors (Tensor): Concatenated tensor after padding the list.
        """
        n_tensors = [len(ts) for ts in tensors]
        #print('n_tensors', n_tensors)
        pad_size = max(n_tensors)

        # pad to has same number of operands for each problem
        tensors = [ts + (pad_size - len(ts)) * [padding]
                for ts in tensors]

        # tensors.size() = (batch_size, pad_size, dim_hidden)
        tensors = torch.stack([torch.stack(t)
                            for t in tensors], dim=0)

        return n_tensors, tensors
