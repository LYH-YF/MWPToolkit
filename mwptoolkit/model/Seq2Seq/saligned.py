import math

import torch
from torch import nn
import numpy as np

from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import SalignedEncoder
from mwptoolkit.module.Decoder.rnn_decoder import SalignedDecoder
from mwptoolkit.module.Environment.stack_machine import OPERATIONS,StackMachine


class Saligned(nn.Module):
    """ Neural Math Word Problem Solver Machine Version 1.

    Args:
        dim_embed (int): Dimension of text embeddings.
        dim_hidden (int): Dimension of encoder decoder hidden state.
    """
    def __init__(self, config,dataset):
        super(Saligned, self).__init__()
        # parameter
        #self._vocab_size = vocab_size = config['vocab_size']
        self._vocab_size = vocab_size = len(dataset.in_word2idx)
        self._dim_embed = dim_embed = config['embedding_size']
        self._dim_hidden = dim_hidden = config['hidden_size']
        self._dropout_rate = dropout_rate = config['dropout_ratio']
        self.NOOP = OPERATIONS.NOOP
        self.GEN_VAR = OPERATIONS.GEN_VAR
        self.ADD = OPERATIONS.ADD
        self.SUB = OPERATIONS.SUB
        self.MUL = OPERATIONS.MUL
        self.DIV = OPERATIONS.DIV
        self.POWER = OPERATIONS.POWER
        self.EQL = OPERATIONS.EQL
        self.N_OPS = OPERATIONS.N_OPS

        max_NUM = list(dataset.out_symbol2idx.keys())[-2]
        self.max_NUM = dataset.out_symbol2idx[max_NUM]
        self.ADD = dataset.out_symbol2idx['+']
        self.POWER = dataset.out_symbol2idx['^']
        self.min_CON = self.N_OPS = self.POWER + 1
        self.UNK = dataset.out_symbol2idx['<UNK>']
        self.max_CON = self.min_NUM - 1
        self.constant = list(dataset.out_symbol2idx.keys())[self.min_CON: self.min_NUM]

        self._device = device = config["device"]
        # module
        #print('vocab_size', config); #exit()
        self.embedder=BaiscEmbedder(vocab_size,
                                    dim_embed,
                                    dropout_rate)
        self.encoder = SalignedEncoder(dim_embed,
                                   dim_hidden,
                                   dim_hidden,
                                   dropout_rate)
        self.decoder = SalignedDecoder(dim_hidden, dropout_rate, device)
        self.embedding_one = nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.embedding_pi = nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))

        # make loss
        class_weights = torch.ones(9)
        class_weights[OPERATIONS.NOOP] = 0
        self._op_loss = nn.CrossEntropyLoss(class_weights,
                                                  size_average=False,
                                                  reduce=False)
        self._arg_loss = nn.CrossEntropyLoss()

    def forward(self, text, ops, text_len, constant_indices, constants, fix_constants, op_len, N_OPS):
        batch_size = len(text)

        # zero embedding for the stack bottom
        bottom = torch.zeros(self._dim_hidden * 2).to(self._device)
        bottom.requires_grad = False

        # deal with device
        seq_emb = self.embedder(text)
        #print('seq_emb', seq_emb.size(), text_len, constant_indices)
        context, state, operands = \
            self.encoder.forward(seq_emb, text_len, constant_indices)
        #print('operands', fix_constants, constants[0], constant_indices);  #exit()
        # initialize stacks
        stacks = [StackMachine(fix_constants + constants[b], operands[b], bottom,
                  dry_run=True)
                  for b in range(batch_size)]

        loss = torch.zeros(batch_size).to(self._device)
        prev_op = torch.zeros(batch_size).to(self._device)
        text_len = text_len.to(self._device)
        prev_output = None

        if True: #self.use_state:
            prev_state = state
        else:
            prev_state = None
        #print('ops', op_len)
        pred_logits = []
        for t in range(max(op_len)):
            # step one
            #print('t', t)
            #if t == 2: exit()
            op_logits, arg_logits, prev_output, prev_state = \
                self.decoder(
                    context, text_len, operands, stacks,
                    prev_op, prev_output, prev_state, self.N_OPS)

            # print('stacks[0]._equations', t, stacks[0]._equations)
            # accumulate op loss
            max_logits = torch.argmax(op_logits, dim=1)
            #max_logits[max_logits == OPERATIONS.N_OPS] = arg_logits
            op_target = ops[:, t].clone().detach()
            op_target[op_target >= self.N_OPS] = self.N_OPS
            op_target.require_grad = False
            #print('op_logits', torch.argmax(op_logits, dim=1), op_target.unsqueeze(0))
            #print('op_logits', op_logits[:5, :5])
            #print('op_target', op_logits.size(), op_target)
            loss += self._op_loss(op_logits, op_target)

            #print(torch.argmax(op_logits, dim=1), op_target)
            #predicts = [stack.get_solution() for stack in stacks]
            #print('predicts', t, predicts)

            # accumulate arg loss
            #print('arg_logits', arg_logits.size(), torch.argmax(arg_logits, dim=1), ops[:, t].unsqueeze(0) - self.N_OPS)
            for b in range(batch_size):
                #print('b', b)
                if ops[b, t] < self.N_OPS:
                    continue
                #print('arg_logits', b, arg_logits[b].size(), ops[b, t].unsqueeze(0) - self.N_OPS)
                #print('stacks[i].stack_log', stacks[b].stack_log)
                loss[b] += self._arg_loss(
                    arg_logits[b].unsqueeze(0),
                    ops[b, t].unsqueeze(0) - self.N_OPS)
            #print(t, prev_op, stacks[0].stack_log_index, stacks[0].stack_log)

            # prev_op = torch.argmax(op_logits, dim=1) #
            prev_op = ops[:, t]
            pred_logits += [torch.argmax(op_logits, dim=1)]

        weights = 1

        loss = (loss * weights).mean()
        pred_logits = torch.stack(pred_logits, 1)
        #print('train pred_logits', pred_logits[0, :])
        predicts = [stack.stack_log_index for stack in stacks]
        #print(pred_logits.size(), ops.size())
        #exit()
        return predicts, loss

    def predict(self, text, ops, text_len, constant_indices, constants, fix_constants, op_len, N_OPS):
        batch_size = len(text)

        # zero embedding for the stack bottom
        bottom = torch.zeros(self._dim_hidden * 2).to(self._device)
        bottom.requires_grad = False

        # deal with device
        seq_emb = self.embedder(text)
        #print('seq_emb', seq_emb.size(), text_len, constant_indices)
        context, state, operands = \
            self.encoder.forward(seq_emb, text_len, constant_indices)
        #print('operands', fix_constants + constants[0], ops); # exit()
        # initialize stacks
        stacks = [StackMachine(fix_constants + constants[b], operands[b], bottom)
                  for b in range(batch_size)]

        loss = torch.zeros(batch_size).to(self._device)
        prev_op = torch.zeros(batch_size).to(self._device)
        prev_output = None
        prev_state = state
        finished = [False] * batch_size
        eql_finished = [False] * batch_size
        pred_logits = []
        for t in range(40):
            op_logits, arg_logits, prev_output, prev_state = \
                self.decoder(
                    context, text_len, operands, stacks,
                    prev_op, prev_output, prev_state, self.N_OPS)

            n_finished = 0
            for b in range(batch_size):
                #print('stacks[b]', stacks[b])
                if stacks[b].get_solution() is not None:
                    finished[b] = True

                if finished[b]:
                    op_logits[b, OPERATIONS.NOOP] = math.inf
                    n_finished += 1
                #print(stacks[b].stack_log_index)
                if (len(stacks[b].stack_log_index) and stacks[b].stack_log_index[-1] == OPERATIONS.EQL):
                    eql_finished[b] = True
                    #print('should break'); exit()

                if stacks[b].get_height() < 2:
                    op_logits[b, OPERATIONS.ADD] = -math.inf
                    op_logits[b, OPERATIONS.SUB] = -math.inf
                    op_logits[b, OPERATIONS.MUL] = -math.inf
                    op_logits[b, OPERATIONS.DIV] = -math.inf
                    op_logits[b, OPERATIONS.POWER] = -math.inf
                    op_logits[b, OPERATIONS.EQL] = -math.inf

            op_loss, prev_op = torch.log(
                torch.nn.functional.softmax(op_logits, -1)
            ).max(-1)
            arg_loss, prev_arg = torch.log(
                torch.nn.functional.softmax(arg_logits, -1)
            ).max(-1)

            for b in range(batch_size):
                if prev_op[b] == OPERATIONS.N_OPS:
                    prev_op[b] += prev_arg[b]
                    loss[b] += arg_loss[b]

                if prev_op[b] != OPERATIONS.NOOP:
                    loss[b] += op_loss[b]

            if n_finished == batch_size:
                break
            if np.sum(eql_finished) == batch_size:
                break #print(stacks[0].stack_log);

        predicts = [None] * batch_size
        #pred_logits = torch.stack(pred_logits, 1)
        # print(pred_logits[:, :5], ops[:, :5])
        for i in range(batch_size):
            #print('stacks[i].stack_log', stacks[i].stack_log_index)
            predicts[i] = {
                'ans': stacks[i].get_solution(),
                'equations': stacks[i].stack_log,
                'equations_index': stacks[i].stack_log_index,
                'confidence': loss[i].item()
            }

        return predicts

    def calculate_loss(self,batch_data):
        text=batch_data["question"]
        ops=batch_data["equation"]
        text_len=batch_data["ques len"]
        constant_indices=batch_data["num pos"]
        constants=batch_data['num list']
        op_len=batch_data["equ len"]
        fix_constants=self.constant
        N_OPS=self.N_OPS
        batch_size = len(text)

        # zero embedding for the stack bottom
        bottom = torch.zeros(self._dim_hidden * 2).to(self._device)
        bottom.requires_grad = False

        # deal with device
        seq_emb = self.embedder(text)
        #print('seq_emb', seq_emb.size(), text_len, constant_indices)
        context, state, operands = \
            self.encoder.forward(seq_emb, text_len, constant_indices)
        #print('operands', fix_constants, constants[0], constant_indices);  #exit()
        # initialize stacks
        stacks = [StackMachine(fix_constants + constants[b], operands[b], bottom,
                  dry_run=True)
                  for b in range(batch_size)]

        loss = torch.zeros(batch_size).to(self._device)
        prev_op = torch.zeros(batch_size).to(self._device)
        text_len = text_len.to(self._device)
        prev_output = None

        if True: #self.use_state:
            prev_state = state
        else:
            prev_state = None
        #print('ops', op_len)
        pred_logits = []
        for t in range(max(op_len)):
            # step one
            #print('t', t)
            #if t == 2: exit()
            op_logits, arg_logits, prev_output, prev_state = \
                self.decoder(
                    context, text_len, operands, stacks,
                    prev_op, prev_output, prev_state, self.N_OPS)

            # print('stacks[0]._equations', t, stacks[0]._equations)
            # accumulate op loss
            max_logits = torch.argmax(op_logits, dim=1)
            #max_logits[max_logits == OPERATIONS.N_OPS] = arg_logits
            op_target = ops[:, t].clone().detach()
            op_target[op_target >= self.N_OPS] = self.N_OPS
            op_target.require_grad = False
            #print('op_logits', torch.argmax(op_logits, dim=1), op_target.unsqueeze(0))
            #print('op_logits', op_logits[:5, :5])
            #print('op_target', op_logits.size(), op_target)
            loss += self._op_loss(op_logits, op_target)

            #print(torch.argmax(op_logits, dim=1), op_target)
            #predicts = [stack.get_solution() for stack in stacks]
            #print('predicts', t, predicts)

            # accumulate arg loss
            #print('arg_logits', arg_logits.size(), torch.argmax(arg_logits, dim=1), ops[:, t].unsqueeze(0) - self.N_OPS)
            for b in range(batch_size):
                #print('b', b)
                if ops[b, t] < self.N_OPS:
                    continue
                #print('arg_logits', b, arg_logits[b].size(), ops[b, t].unsqueeze(0) - self.N_OPS)
                #print('stacks[i].stack_log', stacks[b].stack_log)
                loss[b] += self._arg_loss(
                    arg_logits[b].unsqueeze(0),
                    ops[b, t].unsqueeze(0) - self.N_OPS)
            #print(t, prev_op, stacks[0].stack_log_index, stacks[0].stack_log)

            # prev_op = torch.argmax(op_logits, dim=1) #
            prev_op = ops[:, t]
            pred_logits += [torch.argmax(op_logits, dim=1)]

        weights = 1

        loss = (loss * weights).mean()
        pred_logits = torch.stack(pred_logits, 1)
        #print('train pred_logits', pred_logits[0, :])
        predicts = [stack.stack_log_index for stack in stacks]
        #print(pred_logits.size(), ops.size())
        #exit()
        return predicts, loss