# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:37:56
# @File: saligned.py

import math
from typing import Tuple, Dict, Any

import torch
import numpy as np
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import SalignedEncoder
from mwptoolkit.module.Decoder.rnn_decoder import SalignedDecoder
from mwptoolkit.module.Environment.stack_machine import OPERATIONS, StackMachine
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask, Operators


class Saligned(nn.Module):
    """
    Reference:
        Chiang et al. "Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems".
    """

    def __init__(self, config, dataset):
        super(Saligned, self).__init__()
        self.device = config['device']
        self.operations = operations = OPERATIONS(dataset.out_symbol2idx)
        # parameter
        self._vocab_size = vocab_size = len(dataset.in_idx2word)
        self._dim_embed = dim_embed = config['embedding_size']
        self._dim_hidden = dim_hidden = config['hidden_size']
        self._dropout_rate = dropout_rate = config['dropout_ratio']
        self.max_gen_len = 40
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

        self._device = device = config["device"]

        self.min_NUM = dataset.out_symbol2idx['NUM_0']
        # print(self.dataloader.dataset.out_symbol2idx); exit()
        # self.do_addeql = False if '<BRG>' in dataset.out_symbol2idx else True
        # max_NUM = list(dataset.out_symbol2idx.keys())[-2]
        # self.max_NUM = dataset.out_symbol2idx[max_NUM]
        # self.ADD = dataset.out_symbol2idx['+']
        self.POWER = dataset.out_symbol2idx['^']
        self.min_CON = self.N_OPS_out = self.POWER + 1
        # self.min_CON = self.N_OPS_out = dataset.out_symbol2idx['^']+1 if '<BRG>' not in dataset.out_symbol2idx else dataset.out_symbol2idx['<BRG>']+1
        # self.UNK = dataset.out_symbol2idx['<UNK>']
        # self.max_CON = self.min_NUM - 1
        self.fix_constants = list(dataset.out_symbol2idx.keys())[self.min_CON:self.min_NUM]

        self.mask_list = NumMask.number

        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        # module
        # print('vocab_size', config); #exit()
        self.embedder = BasicEmbedder(vocab_size, dim_embed, dropout_rate)
        self.encoder = SalignedEncoder(dim_embed, dim_hidden, dim_hidden, dropout_rate)
        self.decoder = SalignedDecoder(operations, dim_hidden, dropout_rate, device)
        self.embedding_one = torch.nn.Parameter(torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.embedding_pi = torch.nn.Parameter(torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.encoder.initialize_fix_constant(len(self.fix_constants), self._device)

        # make loss
        class_weights = torch.ones(operations.N_OPS + 1)
        # class_weights[OPERATIONS.NOOP] = 0
        self._op_loss = torch.nn.CrossEntropyLoss(class_weights, size_average=False, reduce=False, ignore_index=-1)
        self._arg_loss = torch.nn.CrossEntropyLoss()

    def forward(self, seq, seq_length, number_list, number_position, number_size, target=None, target_length=None,
                output_all_layers=False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq:
        :param torch.Tensor seq_length:
        :param list number_list:
        :param list number_position:
        :param list number_size:
        :param torch.Tensor | None target:
        :param torch.Tensor | None target_length:
        :param bool output_all_layers:
        :return: token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        constant_indices = number_position
        constants = number_list
        num_len = number_size
        seq_length = seq_length.long()

        batch_size = seq.size(0)
        bottom = torch.zeros(self._dim_hidden * 2).to(self._device)
        bottom.requires_grad = False

        seq_emb = self.embedder(seq)

        encoder_outputs, encoder_hidden, operands, number_emb, encoder_layer_outputs = self.encoder_forward(seq_emb,
                                                                                                            seq_length,
                                                                                                            constant_indices,
                                                                                                            output_all_layers)

        stacks = [StackMachine(self.operations, constants[b] + self.fix_constants, number_emb[b], bottom, dry_run=True)
                  for b in range(batch_size)]

        if target is not None:
            operands_len = torch.LongTensor(self.N_OPS + np.array(num_len)).to(self._device)
            operands_len = operands_len.unsqueeze(1).repeat(1,target.size(1))
            target[(target >= operands_len)] = self.N_OPS

        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden,
                                                                                   seq_length, operands, stacks,
                                                                                   number_emb, target, target_length,
                                                                                   output_all_layers)

        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num pos', 'num list', 'num size'.
        """

        text = torch.tensor(batch_data["question"]).to(self.device)
        ops = torch.tensor(batch_data["equation"]).to(self.device)
        text_len = torch.tensor(batch_data["ques len"]).long()
        ops_len = torch.tensor(batch_data["equ len"]).long()
        constant_indices = batch_data["num pos"]
        constants = batch_data["num list"]
        num_len = batch_data["num size"]

        logits, _, all_layers = self.forward(text, text_len, constants, constant_indices, num_len, ops,
                                                   ops_len, output_all_layers=True)
        (op_logits, arg_logits) = logits
        (op_targets, arg_targets) = all_layers['op_targets'], all_layers['arg_targets']

        batch_size = ops.size(0)
        loss = torch.zeros(batch_size).to(self._device)
        for t in range(max(ops_len)):
            loss += self._op_loss(op_logits[:,t,:], op_targets[:,t])
            for b in range(batch_size):
                if self.NOOP <= arg_targets[b, t] < self.N_OPS:
                    continue
                loss[b] += self._arg_loss(arg_logits[b, t].unsqueeze(0), arg_targets[b, t].unsqueeze(0) - self.N_OPS)

        loss = (loss / max(ops_len)).mean()
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) -> tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num pos', 'num list', 'num size'.
        """
        text = torch.tensor(batch_data['question']).to(self.device)
        text_len = torch.tensor(batch_data['ques len']).long()
        constant_indices = batch_data["num pos"]
        constants = batch_data["num list"]
        num_len = batch_data["num size"]
        target = torch.tensor(batch_data['equation'])

        _, outputs, _ = self.forward(text,text_len,constants,constant_indices,num_len)
        predicts = self.convert_idx2symbol(outputs, constants)
        targets = self.convert_idx2symbol(target, constants)

        return predicts, targets

    def predict(self, batch_data:dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_len = torch.tensor(batch_data["ques len"]).long()
        num_pos = batch_data["num pos"]
        num_list = batch_data["num list"]
        num_size = batch_data["num size"]
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_len, num_list, num_pos, num_size,
                                                                       output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, constant_indices, output_all_layers=False):
        batch_size = seq_emb.size(0)
        encoder_outputs, encoder_hidden, operands = \
            self.encoder.forward(seq_emb, seq_length, constant_indices)
        number_emb = [operands[b_i] + self.encoder.get_fix_constant() for b_i in range(batch_size)]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, operands, number_emb, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, inputs_length, operands, stacks, number_emb, target=None,
                        target_length=None, output_all_layers=False):

        batch_size = encoder_outputs.size(0)
        prev_op = (torch.zeros(batch_size).to(self._device) - 1).type(torch.LongTensor)
        prev_output = None
        prev_state = encoder_hidden

        decoder_outputs = []
        token_logits = []
        arg_logits = []
        outputs = []
        op_targets = []
        arg_targets = []
        if target is not None:
            for t in range(max(target_length)):
                op_logit, arg_logit, prev_output, prev_state = self.decoder(encoder_outputs, inputs_length, operands,
                                                                             stacks, prev_op, prev_output, prev_state,
                                                                             number_emb, self.N_OPS)

                prev_op = target[:, t]
                decoder_outputs.append(prev_output)
                token_logits.append(op_logit)
                arg_logits.append(arg_logit)
                # outputs.append(torch.argmax(op_logits, dim=1))
                op_target = target[:, t].clone().detach()
                op_target[(np.array(target_length) <= t)] = self.NOOP
                op_target[(op_target >= self.N_OPS)] = self.N_OPS
                op_target.require_grad = False
                op_targets.append(op_target)
                _, pred_op = torch.log(torch.nn.functional.softmax(op_logit, -1)).max(-1)
                _, pred_arg = torch.log(torch.nn.functional.softmax(arg_logit, -1)).max(-1)
                for b in range(batch_size):
                    if pred_op[b] == self.N_OPS:
                        pred_op[b] += pred_arg[b]
                outputs.append(pred_op)

        else:
            finished = [False] * batch_size
            for t in range(self.max_gen_len):
                op_logit, arg_logit, prev_output, prev_state = self.decoder(encoder_outputs, inputs_length, operands,
                                                                             stacks, prev_op, prev_output, prev_state,
                                                                             number_emb, self.N_OPS)

                n_finished = 0
                for b in range(batch_size):
                    if len(stacks[b].stack_log_index) and stacks[b].stack_log_index[-1] == self.EQL:
                        finished[b] = True

                    if finished[b]:
                        op_logit[b, self.PAD] = math.inf
                        n_finished += 1

                    # if stacks[b].get_height() < 2:
                    #     op_logit[b, self.ADD] = -math.inf
                    #     op_logit[b, self.SUB] = -math.inf
                    #     op_logit[b, self.MUL] = -math.inf
                    #     op_logit[b, self.DIV] = -math.inf
                    #     op_logit[b, self.POWER] = -math.inf

                op_loss, prev_op = torch.log(torch.nn.functional.softmax(op_logit, -1)).max(-1)
                arg_loss, prev_arg = torch.log(torch.nn.functional.softmax(arg_logit, -1)).max(-1)

                for b in range(batch_size):
                    if prev_op[b] == self.N_OPS:
                        prev_op[b] += prev_arg[b]

                if n_finished == batch_size:
                    break
                decoder_outputs.append(prev_output)
                token_logits.append(op_logit)
                arg_logits.append(arg_logit)
                outputs.append(prev_op)

                if n_finished == batch_size:
                    break

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        token_logits = torch.stack(token_logits, dim=1)
        arg_logits = torch.stack(arg_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        if target is not None:
            op_targets = torch.stack(op_targets,dim=1)
            arg_targets = target.clone()

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['op_logits'] = token_logits
            all_layer_outputs['arg_logits'] = arg_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['op_targets'] = op_targets
            all_layer_outputs['arg_targets'] = arg_targets

        return (token_logits,arg_logits), outputs, all_layer_outputs

    def convert_mask_num(self, batch_output, num_list):
        output_list = []
        for b_i, output in enumerate(batch_output):
            res = []
            num_len = len(num_list[b_i])
            for symbol in output:
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list
