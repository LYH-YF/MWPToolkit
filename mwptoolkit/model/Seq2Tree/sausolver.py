# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:59:55
# @File: sausolver.py

import random
from typing import Dict, Any, Tuple

import torch
from torch import nn
import copy

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Decoder.tree_decoder import SARTreeDecoder
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Layer.tree_layers import Prediction, GenerateNode, Merge, SemanticAlignmentModule
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss, masked_cross_entropy
from mwptoolkit.loss.mse_loss import MSELoss
from mwptoolkit.utils.utils import copy_list
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class SAUSolver(nn.Module):
    """
    Reference:
        Qin et al. "Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems" in EMNLP 2020.
    """

    def __init__(self, config, dataset):
        super(SAUSolver, self).__init__()
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
        self.loss_weight = config['loss_weight']
        self.batch_first = False

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
        # module
        self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        # self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type,
                                       self.dropout_ratio, batch_first=False)
        #self.decoder = SARTreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.decoder = Prediction(self.hidden_size,self.operator_nums,self.generate_size,self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size,
                                           self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.sa = SemanticAlignmentModule(self.hidden_size,self.hidden_size,self.hidden_size)
        self.loss1 = MaskedCrossEntropyLoss()
        #

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

        if not self.batch_first:
            target = target.transpose(0,1)

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

    def calculate_loss(self, batch_data:dict) -> float:
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
        generate_nums = self.generate_nums
        num_start = self.num_start
        # sequence mask for attention
        unk = self.unk_token

        loss = self.train_tree(seq, seq_length, target, target_length, nums_stack, num_size, generate_nums, num_pos, unk, num_start)
        return loss

    def model_test(self, batch_data:dict) -> tuple:
        """Model test.

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
        generate_nums = self.generate_nums
        num_start = self.num_start
        # sequence mask for attention
        all_node_output = self.evaluate_tree(seq, seq_length, generate_nums, num_pos, num_start, self.beam_size,
                                             self.max_out_len)

        all_output = self.convert_idx2symbol(all_node_output, num_list[0], copy_list(nums_stack[0]))
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
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq,seq_length,nums_stack,num_size,num_pos,output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def train_tree(self,input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums, num_pos, unk, num_start, 
                                                   english=False,var_nums=[], batch_first=False):
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums)  # 最大的位置列表数目+常识数字数目+未知数列表
        for i in num_size_batch:
            d = i + len(generate_nums) + len(var_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask)  # 用于屏蔽无关数字，防止生成错误的Nx

        #unk = output_lang.word2index["UNK"]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0, 1)
        target = target_batch.transpose(0, 1)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.decoder.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)

        if self.USE_CUDA:
            input_var = input_var.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        # Zero gradients of both optimizers
        # Run words through encoder

        #encoder_outputs, problem_output = self.encoder(input_var, input_length)
        seq_emb = self.embedder(input_var)
        pade_outputs, _ = self.encoder(seq_emb, input_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

        max_target_length = max(target_length)

        all_node_outputs = []
        all_sa_outputs = []
        # all_leafs = []

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        # 提取与问题相关的数字embedding
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                  self.encoder.hidden_size)

        embeddings_stacks = [[] for _ in range(batch_size)]  # B x 1  当前的tree state/ subtree embedding / output
        left_childs = [None for _ in range(batch_size)]  # B x 1

        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start,
                                                           unk)
            target[t] = target_t
            if self.USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                # 未知数当数字处理，SEP当操作符处理
                if i < num_start:  # 非数字
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                    # print(o[-1].embedding.size())
                    # print(encoder_outputs[idx].size())
                else:  # 数字
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)  # Subtree embedding
                        if batch_first:
                            encoder_mapping, decoder_mapping = self.sa(current_num, encoder_outputs[idx])
                        else:
                            temp_encoder_outputs = encoder_outputs.transpose(0, 1)
                            encoder_mapping, decoder_mapping = self.sa(current_num,temp_encoder_outputs[idx])
                        all_sa_outputs.append((encoder_mapping, decoder_mapping))
                    o.append(TreeEmbedding(current_num, terminal=True))

                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)

                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()  # B x S

        if self.USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            new_all_sa_outputs = []
            for sa_pair in all_sa_outputs:
                new_all_sa_outputs.append((sa_pair[0].cuda(), sa_pair[1].cuda()))
            all_sa_outputs = new_all_sa_outputs
            # target_length = torch.LongTensor(target_length).cuda()
        else:
            pass
            # target_length = torch.LongTensor(target_length)

        semantic_alignment_loss = nn.MSELoss()
        total_semanti_alognment_loss = 0
        sa_len = len(all_sa_outputs)
        for sa_pair in all_sa_outputs:
            total_semanti_alognment_loss += semantic_alignment_loss(sa_pair[0], sa_pair[1])
        # print(total_semanti_alognment_loss)
        total_semanti_alognment_loss = total_semanti_alognment_loss / sa_len
        # print(total_semanti_alognment_loss)

        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss = masked_cross_entropy(all_node_outputs, target,target_length) + 0.01 * total_semanti_alognment_loss
        # loss = loss_0 + loss_1
        loss.backward()
        # clip the grad
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

        # Update parameters with optimizers
        return loss.item()  # , loss_0.item(), loss_1.item()

    def evaluate_tree(self, input_batch, input_length, generate_nums, num_pos, num_start, beam_size=5, max_length=30):

        seq_mask = torch.BoolTensor(1, input_length).fill_(0)
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0, 1)

        num_mask = torch.BoolTensor(1, len(num_pos[0]) + len(generate_nums)).fill_(0)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)

        batch_size = 1

        if self.USE_CUDA:
            input_var = input_var.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()
        # Run words through encoder

        seq_emb = self.embedder(input_var)
        pade_outputs, _ = self.encoder(seq_emb, input_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                       self.hidden_size)
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack,
                                                                                                           left_childs,
                                                                                                           encoder_outputs,
                                                                                                           all_nums_encoder_outputs,
                                                                                                           padding_hidden,
                                                                                                           seq_mask,
                                                                                                           num_mask)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if self.USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
                                                                                  current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        if not self.batch_first:
            encoder_inputs = seq_emb.transpose(0, 1)
        else:
            encoder_inputs = seq_emb
        pade_outputs, hidden_states = self.encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs']=encoder_outputs
            all_layer_outputs['encoder_hidden'] = hidden_states
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output,encoder_outputs,all_layer_outputs

    def decoder_forward(self,encoder_outputs,problem_output,all_nums_encoder_outputs,nums_stack,seq_mask,num_mask,target=None,output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits=[]
        outputs = []
        all_sa_outputs = []
        if target is not None:
            max_target_length = max(target.size(0))
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
                output = torch.topk(token_logit,1,dim=-1)[1]
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
                            if self.batch_first:
                                encoder_mapping, decoder_mapping = self.sa(current_num, encoder_outputs[idx])
                            else:
                                temp_encoder_outputs = encoder_outputs.transpose(0, 1)
                                encoder_mapping, decoder_mapping = self.sa(current_num, temp_encoder_outputs[idx])
                            all_sa_outputs.append((encoder_mapping, decoder_mapping))
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            if not self.batch_first:
                target = target.transpose(0, 1).contiguous()
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
                        current_out.append(torch.squeeze(ti,dim=1))

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
                                     current_left_childs, current_out,current_token_logit))
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
            all_layer_outputs['semantic_alignment_pair']=all_sa_outputs
        return token_logits,outputs,all_layer_outputs

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

    def mse_loss(self, outputs, targets, mask=None):
        # outputs   : [batch_size,output_len,hidden_size]
        # targets   : [batch_size,output_len,hidden_size]
        # mask      : [batch_size,output_len]
        mask = mask.to(self.device)
        x = torch.sqrt(torch.sum(torch.square((outputs - targets)), dim=-1))  # [batch_size,output_len]
        y = torch.sum(x * mask, dim=-1) / torch.sum(mask, dim=-1)  # [batch_size]
        return torch.sum(y)

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

    # def evaluate_tree(self, input_batch, input_length, generate_nums, num_pos, num_start, beam_size=5, max_length=30,var_nums=[]):
    #     # sequence mask for attention
    #     seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    #     # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    #     input_var = torch.LongTensor(input_batch).unsqueeze(1)
    #
    #     num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)
    #
    #     # Set to not-training mode to disable dropout
    #
    #     padding_hidden = torch.FloatTensor([0.0 for _ in range(self.decoder.hidden_size)]).unsqueeze(0)
    #
    #     batch_size = 1
    #
    #     if self.USE_CUDA:
    #         input_var = input_var.cuda()
    #         seq_mask = seq_mask.cuda()
    #         padding_hidden = padding_hidden.cuda()
    #         num_mask = num_mask.cuda()
    #
    #     # Run words through encoder
    #     encoder_outputs, problem_output = self.encoder(input_var, input_length)
    #
    #     # Prepare input and output variables  # # root embedding B x 1
    #     node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    #
    #     num_size = len(num_pos)
    #     # 提取与问题相关的数字embedding
    #     all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
    #                                                               self.encoder.hidden_size)
    #     # B x P x N
    #     embeddings_stacks = [[] for _ in range(batch_size)]
    #     left_childs = [None for _ in range(batch_size)]
    #     beam_search=True
    #     if beam_search:
    #         beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    #
    #         for t in range(max_length):
    #             current_beams = []
    #             while len(beams) > 0:
    #                 b = beams.pop()
    #                 if len(b.node_stack[0]) == 0:
    #                     current_beams.append(b)
    #                     continue
    #                 # left_childs = torch.stack(b.left_childs)
    #                 left_childs = b.left_childs
    #
    #                 num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(
    #                     b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
    #                     seq_mask, num_mask)
    #
    #                 # leaf = p_leaf[:, 0].unsqueeze(1)
    #                 # repeat_dims = [1] * leaf.dim()
    #                 # repeat_dims[1] = op.size(1)
    #                 # leaf = leaf.repeat(*repeat_dims)
    #                 #
    #                 # non_leaf = p_leaf[:, 1].unsqueeze(1)
    #                 # repeat_dims = [1] * non_leaf.dim()
    #                 # repeat_dims[1] = num_score.size(1)
    #                 # non_leaf = non_leaf.repeat(*repeat_dims)
    #                 #
    #                 # p_leaf = torch.cat((leaf, non_leaf), dim=1)
    #                 out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
    #
    #                 # out_score = p_leaf * out_score
    #
    #                 topv, topi = out_score.topk(beam_size)
    #
    #                 # is_leaf = int(topi[0])
    #                 # if is_leaf:
    #                 #     topv, topi = op.topk(1)
    #                 #     out_token = int(topi[0])
    #                 # else:
    #                 #     topv, topi = num_score.topk(1)
    #                 #     out_token = int(topi[0]) + num_start
    #                 for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
    #                     current_node_stack = copy_list(b.node_stack)
    #                     current_left_childs = []
    #                     current_embeddings_stacks = copy_list(b.embedding_stack)
    #                     current_out = copy.deepcopy(b.out)
    #                     out_token = int(ti)
    #                     current_out.append(out_token)
    #
    #                     node = current_node_stack[0].pop()
    #
    #                     # var_num当时数字处理，SEP/;当操作符处理
    #                     if out_token < num_start:  # 非数字
    #                         generate_input = torch.LongTensor([out_token])
    #                         if self.USE_CUDA:
    #                             generate_input = generate_input.cuda()
    #                         left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
    #                                                                        current_context)
    #
    #                         current_node_stack[0].append(TreeNode(right_child))
    #                         current_node_stack[0].append(TreeNode(left_child, left_flag=True))
    #
    #                         current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
    #                     else:  # 数字
    #                         current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
    #
    #                         while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                             sub_stree = current_embeddings_stacks[0].pop()
    #                             op = current_embeddings_stacks[0].pop()
    #                             current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
    #                         current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
    #                     if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                         current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
    #                     else:
    #                         current_left_childs.append(None)
    #                     current_beams.append(
    #                         TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
    #                                  current_left_childs, current_out))
    #             beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
    #             beams = beams[:beam_size]
    #             flag = True
    #             for b in beams:
    #                 if len(b.node_stack[0]) != 0:
    #                     flag = False
    #             if flag:
    #                 break
    #
    #         return beams[0].out
    #     else:
    #         all_node_outputs = []
    #         for t in range(max_length):
    #             num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(
    #                 node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
    #                 seq_mask, num_mask)
    #
    #             out_scores = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
    #             out_tokens = torch.argmax(out_scores, dim=1)  # B
    #             all_node_outputs.append(out_tokens)
    #             left_childs = []
    #             for idx, node_stack, out_token, embeddings_stack in zip(range(batch_size), node_stacks, out_tokens,
    #                                                                     embeddings_stacks):
    #                 # node = node_stack.pop()
    #                 if len(node_stack) != 0:
    #                     node = node_stack.pop()
    #                 else:
    #                     left_childs.append(None)
    #                     continue
    #                 # var_num当时数字处理，SEP/;当操作符处理
    #                 if out_token < num_start:  # 非数字
    #                     generate_input = torch.LongTensor([out_token])
    #                     if self.USE_CUDA:
    #                         generate_input = generate_input.cuda()
    #                     left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
    #                                                                    current_context)
    #                     node_stack.append(TreeNode(right_child))
    #                     node_stack.append(TreeNode(left_child, left_flag=True))
    #                     embeddings_stack.append(TreeEmbedding(node_label.unsqueeze(0), False))
    #                 else:  # 数字
    #                     current_num = current_nums_embeddings[idx, out_token - num_start].unsqueeze(0)
    #                     while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
    #                         sub_stree = embeddings_stack.pop()
    #                         op = embeddings_stack.pop()
    #                         current_num = self.merge(op.embedding.squeeze(0), sub_stree.embedding, current_num)
    #                     embeddings_stack.append(TreeEmbedding(current_num, terminal=True))
    #
    #                 if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
    #                     left_childs.append(embeddings_stack[-1].embedding)
    #                 else:
    #                     left_childs.append(None)
    #
    #         # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    #         all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    #         all_node_outputs = all_node_outputs.cpu().numpy()
    #         return all_node_outputs[0]