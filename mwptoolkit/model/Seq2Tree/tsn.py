# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 05:00:56
# @File: tsn.py

import copy
import itertools

import torch
from torch import nn
from typing import Tuple, Dict, Any

from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Layer.tree_layers import Prediction, GenerateNode, Merge
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss, masked_cross_entropy
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask
from mwptoolkit.utils.utils import str2float, copy_list, clones


class TSN(nn.Module):
    """
    Reference:
        Zhang et al. "Teacher-Student Networks with Multiple Decoders for Solving Math Word Problem" in IJCAI 2020.
    """
    def __init__(self, config, dataset):
        super(TSN, self).__init__()
        # parameter
        self.hidden_size = config["hidden_size"]
        self.bidirectional = config["bidirectional"]
        self.device = config["device"]
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.alpha = 0.15
        #self.max_input_len=config['max_len']
        self.max_encoder_mask_len = config['max_encoder_mask_len']
        if self.max_encoder_mask_len == None:
            self.max_encoder_mask_len = 128

        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
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

        self.t_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        #self.t_encoder = GraphBasedEncoder(self.embedding_size,self.hidden_size,self.num_layers,self.dropout_ratio)
        self.t_decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.t_node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.t_merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.s_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        #self.s_encoder = GraphBasedEncoder(self.embedding_size,self.hidden_size, self.num_layers,self.dropout_ratio)
        self.s_decoder_1 = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_1 = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_1 = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_decoder_2 = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_2 = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_2 = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()
        self.soft_target = {}

    def forward(self,seq, seq_length, nums_stack, num_size, num_pos, target=None,output_all_layers=False):
        """

        :param seq:
        :param seq_length:
        :param nums_stack:
        :param num_size:
        :param num_pos:
        :param target:
        :param output_all_layers:
        :return:
        """
        t_token_logits, t_symbol_outputs, t_net_all_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target=target,output_all_layers=output_all_layers)
        s_token_logits, s_symbol_outputs, s_net_all_outputs = self.student_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target=target,output_all_layers=output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(t_net_all_outputs)
            model_all_outputs.update(s_net_all_outputs)
            model_all_outputs['soft_target'] = t_token_logits.clone().detach()

        return (t_token_logits,s_token_logits[0],s_token_logits[1]),(t_symbol_outputs,s_symbol_outputs[0],s_symbol_outputs[1]),model_all_outputs

    def teacher_net_forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None,output_all_layers=False)\
            -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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
        seq_emb = self.t_embedder(seq)

        problem_output,encoder_outputs,encoder_layer_outputs = self.teacher_net_encoder_forward(seq_emb,seq_length,output_all_layers)

        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size,
                                                                       max_num_size,
                                                                       self.hidden_size)

        token_logits, symbol_outputs, decoder_layer_outputs = self.teacher_net_decoder_forward(encoder_outputs, problem_output,
                                                                                   all_nums_encoder_outputs, nums_stack,
                                                                                   seq_mask, num_mask, target,
                                                                                   output_all_layers)
        teacher_net_all_outputs = {}
        if output_all_layers:
            teacher_net_all_outputs['teacher_inputs_embedding'] = seq_emb
            teacher_net_all_outputs.update(encoder_layer_outputs)
            teacher_net_all_outputs['teacher_number_representation'] = all_nums_encoder_outputs
            teacher_net_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, teacher_net_all_outputs

    def student_net_forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None,output_all_layers=False)\
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:(token_logits_1,token_logits_2), symbol_outputs:(symbol_outputs_1,symbol_outputs_2), model_all_outputs.
        :rtype: tuple(tuple(torch.Tensor), tuple(torch.Tensor), dict)
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
        seq_emb = self.t_embedder(seq)

        problem_output, encoder_outputs, encoder_layer_outputs = self.student_net_encoder_forward(seq_emb, seq_length,
                                                                                                  output_all_layers)

        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size,
                                                                       max_num_size,
                                                                       self.hidden_size)

        token_logits, symbol_outputs, decoder_layer_outputs = self.student_net_decoder_forward(encoder_outputs,
                                                                                               problem_output,
                                                                                               all_nums_encoder_outputs,
                                                                                               nums_stack,
                                                                                               seq_mask, num_mask,
                                                                                               target,
                                                                                               output_all_layers)
        student_net_all_outputs = {}
        if output_all_layers:
            student_net_all_outputs['student_inputs_embedding'] = seq_emb
            student_net_all_outputs.update(encoder_layer_outputs)
            student_net_all_outputs['student_number_representation'] = all_nums_encoder_outputs
            student_net_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, student_net_all_outputs

    def teacher_calculate_loss(self, batch_data:dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation of teacher net.

        :param batch_data: one batch data.
        :return: loss value

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

        token_logits, _, t_net_layer_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos,
                                                                        target, output_all_layers=True)
        target = t_net_layer_outputs['teacher_target']

        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def student_calculate_loss(self, batch_data:dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation of student net.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos', 'id'
        """

        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        target_length = torch.LongTensor(batch_data["equ len"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]

        batch_id = batch_data["id"]

        soft_target = self.get_soft_target(batch_id)
        soft_target = torch.cat(soft_target, dim=0).to(self.device)

        token_logits,_,s_net_layer_outputs = self.student_net_forward(seq,seq_length,nums_stack,num_size,num_pos,target,output_all_layers=True)

        (token_logits_1, token_logits_2) = token_logits
        target1 = s_net_layer_outputs['student_1_target']
        target2 = s_net_layer_outputs['student_2_target']
        loss1 = masked_cross_entropy(token_logits_1, target1, target_length)
        loss2 = soft_target_loss(token_logits_1, soft_target, target_length)
        loss3 = masked_cross_entropy(token_logits_2, target2, target_length)
        loss4 = soft_target_loss(token_logits_2, soft_target, target_length)
        cos_loss = cosine_loss(token_logits_1, token_logits_2, target_length)

        loss = 0.85 * loss1 + 0.15 * loss2 + 0.85 * loss3 + 0.15 * loss4 + 0.1 * cos_loss
        loss.backward()

        return loss.item()

    def teacher_test(self, batch_data:dict) -> tuple:
        """Teacher net test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """

        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_pos = batch_data["num pos"]
        num_list = batch_data['num list']
        num_size = batch_data['num size']

        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)

        all_output = self.convert_idx2symbol(outputs, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_output, targets

    def student_test(self, batch_data:dict) -> Tuple[list, float, list, float, list]:
        """Student net test.

        :param batch_data: one batch data.
        :return: predicted equation1, score1, predicted equation2, score2, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """

        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_pos = batch_data["num pos"]
        num_list = batch_data['num list']
        num_size = batch_data['num size']

        _,outputs,s_net_layer_outputs = self.student_net_forward(seq,seq_length,nums_stack,num_size,num_pos,output_all_layers=True)

        (outputs_1,outputs_2) = outputs
        score1 = s_net_layer_outputs['student_1_score']
        score2 = s_net_layer_outputs['student_2_score']
        all_output1 = self.convert_idx2symbol(outputs_1, num_list[0], copy_list(nums_stack[0]))
        all_output2 = self.convert_idx2symbol(outputs_2, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_output1, score1, all_output2, score2, targets

    def model_test(self, batch_data):
        return

    def predict(self, batch_data:dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        raise NotImplementedError

    def teacher_net_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.t_encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['teacher_encoder_outputs'] = encoder_outputs
            all_layer_outputs['teacher_encoder_hidden'] = hidden_states
            all_layer_outputs['teacher_inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def teacher_net_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack,
                                    seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(
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
                left_child, right_child, node_label = self.t_node_generater(current_embeddings, generate_input,
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
                            current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
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

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(
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
                            left_child, right_child, node_label = self.t_node_generater(current_embeddings,
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
                                current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
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
            all_layer_outputs['teacher_token_logits'] = token_logits
            all_layer_outputs['teacher_outputs'] = outputs
            all_layer_outputs['teacher_target'] = target
        return token_logits, outputs, all_layer_outputs

    def student_net_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.s_encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_encoder_outputs'] = encoder_outputs
            all_layer_outputs['student_encoder_hidden'] = hidden_states
            all_layer_outputs['student_inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def student_net_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack,
                                    seq_mask, num_mask, target=None, output_all_layers=False):
        s_1_token_logits, s_1_outputs, s_1_all_layer_outputs = self.student_net_1_decoder_forward(encoder_outputs,
                                                                                                  problem_output,
                                                                                                  all_nums_encoder_outputs,
                                                                                                  nums_stack,
                                                                                                  seq_mask, num_mask,
                                                                                                  target=target,
                                                                                                  output_all_layers=output_all_layers)
        s_2_token_logits, s_2_outputs, s_2_all_layer_outputs = self.student_net_2_decoder_forward(encoder_outputs,
                                                                                                  problem_output,
                                                                                                  all_nums_encoder_outputs,
                                                                                                  nums_stack,
                                                                                                  seq_mask, num_mask,
                                                                                                  target=target,
                                                                                                  output_all_layers=output_all_layers)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs.update(s_1_all_layer_outputs)
            all_layer_outputs.update(s_2_all_layer_outputs)

        return (s_1_token_logits, s_2_token_logits), (s_1_outputs, s_2_outputs), all_layer_outputs

    def student_net_1_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack,
                                      seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        score = None
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(
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
                left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input,
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
                            current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
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

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(
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
                            left_child, right_child, node_label = self.s_node_generater_1(current_embeddings,
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
                                current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
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
            score = beams[0].score
        token_logits = torch.stack(token_logits, dim=1)  # B x S x N
        outputs = torch.stack(outputs, dim=1)  # B x S
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_1_token_logits'] = token_logits
            all_layer_outputs['student_1_outputs'] = outputs
            all_layer_outputs['student_1_target'] = target
            all_layer_outputs['student_1_score'] = score
        return token_logits, outputs, all_layer_outputs

    def student_net_2_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack,
                                      seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        seq_size = encoder_outputs.size(0)
        encoder_outputs_mask = self.encoder_mask[:batch_size, :seq_size, :].transpose(1, 0).float()
        encoder_outputs = encoder_outputs * encoder_outputs_mask.float()

        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        score = None
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(
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
                left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input,
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
                            current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
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

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(
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
                            left_child, right_child, node_label = self.s_node_generater_1(current_embeddings,
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
                                current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
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
            score = beams[0].score
        token_logits = torch.stack(token_logits, dim=1)  # B x S x N
        outputs = torch.stack(outputs, dim=1)  # B x S
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_2_token_logits'] = token_logits
            all_layer_outputs['student_2_outputs'] = outputs
            all_layer_outputs['student_2_target'] = target
            all_layer_outputs['student_2_score'] = score
        return token_logits, outputs, all_layer_outputs

    def build_graph(self, seq_length, num_list, num_pos, group_nums):
        max_len = seq_length.max()
        batch_size = len(seq_length)
        batch_graph = []
        for b_i in range(batch_size):
            x = torch.zeros((max_len, max_len))
            for idx in range(seq_length[b_i]):
                x[idx, idx] = 1
            quantity_cell_graph = torch.clone(x)
            graph_greater = torch.clone(x)
            graph_lower = torch.clone(x)
            graph_quanbet = torch.clone(x)
            graph_attbet = torch.clone(x)
            for idx, n_pos in enumerate(num_pos[b_i]):
                for pos in group_nums[b_i][idx]:
                    quantity_cell_graph[n_pos, pos] = 1
                    quantity_cell_graph[pos, n_pos] = 1
                    graph_quanbet[n_pos, pos] = 1
                    graph_quanbet[pos, n_pos] = 1
                    graph_attbet[n_pos, pos] = 1
                    graph_attbet[pos, n_pos] = 1
            for idx_i in range(len(num_pos[b_i])):
                for idx_j in range(len(num_pos[b_i])):
                    num_i = str2float(num_list[b_i][idx_i])
                    num_j = str2float(num_list[b_i][idx_j])

                    if num_i > num_j:
                        graph_greater[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
                        graph_lower[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                    else:
                        graph_greater[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                        graph_lower[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
            group_num_ = itertools.chain.from_iterable(group_nums[b_i])
            combn = itertools.permutations(group_num_, 2)
            for idx in combn:
                graph_quanbet[idx] = 1
                graph_quanbet[idx] = 1
                graph_attbet[idx] = 1
                graph_attbet[idx] = 1
            quantity_cell_graph = quantity_cell_graph.to(self.device)
            graph_greater = graph_greater.to(self.device)
            graph_lower = graph_lower.to(self.device)
            graph_quanbet = graph_quanbet.to(self.device)
            graph_attbet = graph_attbet.to(self.device)
            graph = torch.stack([quantity_cell_graph, graph_greater, graph_lower, graph_quanbet, graph_attbet], dim=0)
            batch_graph.append(graph)
        batch_graph = torch.stack(batch_graph)
        return batch_graph

    def init_encoder_mask(self, batch_size):
        encoder_mask = torch.FloatTensor(batch_size, self.max_encoder_mask_len, self.hidden_size).uniform_() < 0.99
        self.encoder_mask = encoder_mask.float().to(self.device)

    @torch.no_grad()
    def init_soft_target(self, batch_data):
        """Build soft target
        
        Args:
            batch_data (dict): one batch data.
        
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        target_length = torch.tensor(batch_data["equ len"]).to(self.device)

        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]

        ques_id = batch_data["id"]
        all_node_outputs, _, t_net_layer_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos,
                                                                        target)
        all_node_outputs = all_node_outputs.cpu()
        for id_, soft_target in zip(ques_id, all_node_outputs.split(1)):
            self.soft_target[id_] = soft_target

    def get_soft_target(self, batch_id):
        soft_tsrget = []
        for id_ in batch_id:
            soft_tsrget.append(self.soft_target[id_])
        return soft_tsrget

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

    def convert_idx2symbol(self, output, num_list, num_stack):
        #batch_size=output.size(0)
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


def soft_target_loss(logits, soft_target, length):
    loss_total = []
    for predict, label in zip(logits.split(1, dim=1), soft_target.split(1, dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = soft_cross_entropy_loss(predict, label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total, dim=0).transpose(1, 0)
    #loss_total = loss_total.sum(dim=1)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total


def soft_cross_entropy_loss(predict_score, label_score):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    softmax = torch.nn.Softmax(dim=1)

    predict_prob_log = log_softmax(predict_score).float()
    label_prob = softmax(label_score).float()

    loss_elem = -label_prob * predict_prob_log
    loss = loss_elem.sum(dim=1)
    return loss


def cosine_loss(logits, logits_1, length):
    loss_total = []
    for predict, label in zip(logits.split(1, dim=1), logits_1.split(1, dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = cosine_sim(predict, label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total, dim=0).transpose(1, 0)
    #loss_total = loss_total.sum(dim=1)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total


def cosine_sim(logits, logits_1):
    device = logits.device
    return torch.ones(logits.size(0)).to(device) + torch.cosine_similarity(logits, logits_1, dim=1).to(device)
