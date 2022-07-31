# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:59:38
# @File: gts.py

import copy
from typing import Tuple, Any, Dict

import torch

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Embedder.roberta_embedder import RobertaEmbedder
from mwptoolkit.module.Embedder.bert_embedder import BertEmbedder
from mwptoolkit.module.Layer.tree_layers import *
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.module.Strategy.weakly_supervising import out_expression_list
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss, masked_cross_entropy
from mwptoolkit.utils.utils import copy_list, get_weakly_supervised
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class GTS(nn.Module):
    """
    Reference:
        Xie et al. "A Goal-Driven Tree-Structured Neural Model for Math Word Problems" in IJCAI 2019.
    """

    def __init__(self, config, dataset):
        super(GTS, self).__init__()
        # parameter
        self.hidden_size = config["hidden_size"]
        self.device = config["device"]
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.embedding = config['embedding']

        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.in_word2idx = dataset.in_word2idx
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)

        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
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
        try:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.in_pad_token = None
        # module
        if config['embedding'] == 'roberta':
            self.embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
        elif config['embedding'] == 'bert':
            self.embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
        else:
            self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        # self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type,
                                       self.dropout_ratio, batch_first=False)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size,
                                           self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) -> Tuple[
            torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        # sequence mask for attention
        seq_mask = torch.eq(seq, self.in_pad_token).to(self.device)

        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        batch_size = len(seq_length)

        seq_emb = self.embedder(seq)

        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length,
                                                                                      output_all_layers)

        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size,
                                                                       max_num_size,
                                                                       self.hidden_size)

        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output,
                                                                                   all_nums_encoder_outputs, nums_stack,
                                                                                   seq_mask, num_mask, target,
                                                                                   output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        target_length = torch.LongTensor(batch_data["equ len"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]

        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, target,
                                                          output_all_layers=True)
        target = all_layer_outputs['target']

        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) -> tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list','num size'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_pos = batch_data["num pos"]
        num_list = batch_data['num list']
        num_size = batch_data['num size']

        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)

        all_output = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self,batch_data:dict,output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos,
                                                                       output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices.cuda()
            masked_index = masked_index.cuda()
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        # when the decoder input is copied num but the num has two pos, chose the max
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = hidden_states
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask,
                        target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(
                    node_stacks,
                    left_childs,
                    encoder_outputs,
                    all_nums_encoder_outputs,
                    padding_hidden,
                    seq_mask,
                    num_mask)

                # all_leafs.append(p_leaf)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)

                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack,
                                                                    self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
                                                                          current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                       node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue

                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(
                        b.node_stack,
                        left_childs,
                        encoder_outputs,
                        all_nums_encoder_outputs,
                        padding_hidden,
                        seq_mask,
                        num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)

                    out_score = nn.functional.log_softmax(token_logit, dim=1)

                    # out_score = p_leaf * out_score

                    topv, topi = out_score.topk(self.beam_size)

                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]

                        current_token_logit.append(token_logit)

                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))

                        node = current_node_stack[0].pop()

                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input.cuda()
                            left_child, right_child, node_label = self.node_generater(current_embeddings,
                                                                                      generate_input,
                                                                                      current_context)

                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)

                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(
                            TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                     current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)  # B x S x N
        outputs = torch.stack(outputs, dim=1)  # B x S
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def convert_idx2symbol(self, output, num_list, num_stack):
        # batch_size=output.size(0)
        '''batch_size=1'''
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list

    # def evaluate_tree_batch(self, input_batch, input_length, generate_nums, num_pos, num_start, beam_size=5,
    #                         max_length=30):
    #
    #     batch_size = input_batch.size(0)
    #     seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    #     # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    #     input_var = input_batch.transpose(0, 1)
    #
    #     num_mask = torch.BoolTensor(1, len(num_pos[0]) + len(generate_nums)).fill_(0)
    #
    #     padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
    #
    #     if self.USE_CUDA:
    #         input_var = input_var.cuda()
    #         seq_mask = seq_mask.cuda()
    #         padding_hidden = padding_hidden.cuda()
    #         num_mask = num_mask.cuda()
    #     # Run words through encoder
    #     if self.embedding == 'roberta':
    #         seq_emb = self.embedder(input_var, seq_mask)
    #     else:
    #         seq_emb = self.embedder(input_var)
    #     # seq_emb = self.embedder(input_var)
    #     pade_outputs, _ = self.encoder(seq_emb, input_length)
    #     problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
    #     encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
    #
    #     # Prepare input and output variables
    #     node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    #
    #     num_size = len(num_pos[0])
    #     all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
    #                                                                    self.hidden_size)
    #     # B x P x N
    #     embeddings_stacks = [[] for _ in range(batch_size)]
    #     left_childs = [None for _ in range(batch_size)]
    #
    #     beams = [TreeBeam(0.0, node_stacks[b_i], embeddings_stacks[b_i], left_childs[b_i], []) for b_i in
    #              range(batch_size)]
    #
    #     for t in range(max_length):
    #         current_beams = [[] for _ in range(batch_size)]
    #         node_stack_batch = [None for _ in range(batch_size)]
    #         current_beam = [None for _ in range(batch_size)]
    #         for b_i in range(batch_size):
    #             flag_i = False
    #             while len(beams[b_i]) > 0:
    #                 b = beams[b_i].pop()
    #                 if len(b.node_stack[0]) == 0:
    #                     current_beams[b_i].append(b)
    #                     continue
    #                 current_beam[b_i] = b
    #                 left_childs[b_i] = b.left_childs
    #                 node_stack_batch[b_i] = b.node_stack
    #                 flag_i = True
    #                 break
    #             if not flag_i:
    #                 left_childs[b_i] = padding_hidden
    #                 node_stack_batch[b_i] = padding_hidden
    #
    #         node_stack_batch = torch.stack(node_stack_batch)
    #         num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stack_batch,
    #                                                                                                    left_childs,
    #                                                                                                    encoder_outputs,
    #                                                                                                    all_nums_encoder_outputs,
    #                                                                                                    padding_hidden,
    #                                                                                                    seq_mask,
    #                                                                                                    num_mask)
    #
    #         out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
    #
    #         # out_score = p_leaf * out_score
    #
    #         topv, topi = out_score.topk(beam_size)
    #
    #         for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
    #             for b_i in range(batch_size):
    #                 current_node_stack = copy_list(node_stack_batch[b_i])
    #                 current_left_childs = []
    #                 current_embeddings_stacks = copy_list(current_beam[b_i].embedding_stack)
    #                 current_out = copy.deepcopy(current_beam[b_i].out)
    #
    #                 out_token = int(ti[b_i])
    #                 current_out.append(out_token)
    #
    #                 node = current_node_stack[0].pop()
    #
    #                 if out_token < num_start:
    #                     generate_input = torch.LongTensor([out_token])
    #                     if self.USE_CUDA:
    #                         generate_input = generate_input.cuda()
    #                     left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
    #                                                                               current_context)
    #
    #                     current_node_stack[0].append(TreeNode(right_child))
    #                     current_node_stack[0].append(TreeNode(left_child, left_flag=True))
    #
    #                     current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
    #                 else:
    #                     current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
    #
    #                     while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                         sub_stree = current_embeddings_stacks[0].pop()
    #                         op = current_embeddings_stacks[0].pop()
    #                         current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
    #                     current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
    #                 if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                     current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
    #                 else:
    #                     current_left_childs.append(None)
    #                 current_beams[b_i].append(
    #                     TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
    #                              current_left_childs, current_out))
    #             beams = [sorted(current_beam, key=lambda x: x.score, reverse=True) for current_beam in current_beams]
    #             beams = [beam[:beam_size] for beam in beams]
    #             flag = [True for _ in range(batch_size)]
    #             for b_i in range(batch_size):
    #                 for b in beams[b_i]:
    #                     if len(b.node_stack[0]) != 0:
    #                         flag[b_i] = False
    #             if flag.count(True) == batch_size:
    #                 break
    #
    #     return [beams[b_i].out for b_i in range(batch_size)]

