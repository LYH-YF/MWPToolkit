# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 05:00:12
# @File: treelstm.py

import copy
from typing import Tuple, Dict, Any

import torch
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Decoder.tree_decoder import LSTMBasedTreeDecoder
from mwptoolkit.module.Layer.tree_layers import NodeEmbeddingLayer, TreeNode, TreeEmbedding
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask
from mwptoolkit.utils.utils import copy_list


class TreeLSTM(nn.Module):
    """
    Reference:
        Liu et al. "Tree-structured Decoding for Solving Math Word Problems" in EMNLP | IJCNLP 2019.
    """

    def __init__(self, config, dataset):
        super(TreeLSTM, self).__init__()
        # parameter
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.num_layers = config["num_layers"]
        self.dropout_ratio = config['dropout_ratio']

        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)

        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.mask_list = NumMask.number
        self.operator_nums = dataset.operator_nums
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.generate_size = len(generate_list)
        self.num_start = dataset.num_start
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        # self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]

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
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, \
                               batch_first=True, dropout=self.dropout_ratio, bidirectional=True)
        self.decoder = LSTMBasedTreeDecoder(self.embedding_size, self.hidden_size * self.num_layers, self.operator_nums,
                                            self.generate_size, self.dropout_ratio)
        self.node_generater = NodeEmbeddingLayer(self.operator_nums, self.embedding_size)
        self.root = nn.Parameter(torch.randn(1, self.embedding_size))

        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None,
                output_all_layers=False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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
        seq_mask = []
        max_len = max(seq_length)
        for i in seq_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        seq_emb = self.embedder(seq)

        encoder_outputs, initial_hidden, problem_output, encoder_layer_outputs = self.encoder_forward(seq_emb,
                                                                                                      output_all_layers=output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size,
                                                                       self.hidden_size * self.num_layers)

        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, initial_hidden,
                                                                                   problem_output,
                                                                                   all_nums_encoder_outputs, seq_mask,
                                                                                   num_mask, nums_stack, target,
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
        # target_length = torch.LongTensor(target_length).to(self.device)
        self.loss.reset()
        self.loss.eval_batch(token_logits, target, target_length)
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) -> tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', num size, 'num list'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_pos = batch_data["num pos"]
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_outputs = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_outputs, targets

    def predict(self, batch_data:dict, output_all_layers=False):
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

    def encoder_forward(self, seq_emb, output_all_layers=False):
        pade_outputs, initial_hidden = self.encoder(seq_emb)
        problem_output = torch.cat([pade_outputs[:, -1, :self.hidden_size], pade_outputs[:, 0, self.hidden_size:]],
                                   dim=1)
        encoder_outputs = pade_outputs

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = initial_hidden
            all_layer_outputs['inputs_representation'] = problem_output

        return encoder_outputs, initial_hidden, problem_output, all_layer_outputs

    def decoder_forward(self, encoder_outputs, initial_hidden, problem_output, all_nums_encoder_outputs, seq_mask,
                        num_mask, nums_stack, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.embedding_size)]).unsqueeze(0).to(self.device)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        all_node_outputs = []
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        nodes = [[] for _ in range(batch_size)]

        hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1),
                  initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))
        tree_hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1),
                       initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))

        nodes_hiddens = [[] for _ in range(batch_size)]

        parent = [hidden[0][idx].unsqueeze(0) for idx in range(batch_size)]
        left = [self.root for _ in range(batch_size)]
        prev = [self.root for _ in range(batch_size)]

        token_logits = []
        outputs = []

        if target is not None:
            max_target_length = target.size(1)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(
                    parent, left, prev, encoder_outputs, all_nums_encoder_outputs,
                    padding_hidden, seq_mask, num_mask, hidden, tree_hidden)

                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)

                target_t, generate_input = self.generate_tree_input(target[:, t].tolist(), outputs, nums_stack,
                                                                    self.num_start, self.unk_token)
                target[:, t] = target_t
                generate_input = generate_input.to(self.device)
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input,
                                                                          current_context)
                # print("left_child, right_child, node_label", left_child, right_child, node_label); exit()

                left, parent, prev, prev_idx = [], [], [], []
                left_childs = []
                for idx, l, r, node_stack, i, o, n, n_hidden in zip(range(batch_size), left_child.split(1),
                                                                    right_child.split(1), node_stacks,
                                                                    target[:, t].tolist(), embeddings_stacks, nodes,
                                                                    nodes_hiddens):
                    continue_flag = False
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue_flag = True
                        # continue

                    if not continue_flag:
                        if i < self.num_start:
                            n.append(i)
                            n_hidden.append(hidden[0][idx].unsqueeze(0))
                            node_stack.append(TreeNode(r))
                            node_stack.append(TreeNode(l, left_flag=True))
                            o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                            while len(o) > 0 and o[-1].terminal:
                                sub_stree = o.pop()
                                op = o.pop()
                                n.pop()
                                n.pop()
                                n_hidden.pop()
                                n_hidden.pop()
                                current_num = sub_stree.embedding
                            o.append(TreeEmbedding(current_num, True))
                            n.append(i)
                            n_hidden.append(hidden[0][idx].unsqueeze(0))
                        if len(o) > 0 and o[-1].terminal:
                            left_childs.append(o[-1].embedding)
                        else:
                            left_childs.append(None)

                    parent_flag = True
                    if len(node_stack) == 0:
                        left.append(self.root)
                        parent.append(hidden[0][idx].unsqueeze(0))
                        prev.append(self.root)
                        prev_idx.append(None)
                    elif n[-1] < self.num_start:
                        left.append(self.root)
                        parent.append(n_hidden[-1])
                        prev.append(self.node_generater.embeddings(torch.LongTensor([n[-1]]).to(self.device)))
                        prev_idx.append(n[-1])
                    else:
                        left.append(current_nums_embeddings[idx, n[-1] - self.num_start].unsqueeze(0))
                        prev.append(current_nums_embeddings[idx, n[-1] - self.num_start].unsqueeze(0))
                        for i in range(len(n) - 1, -1, -1):
                            if n[i] < self.num_start:
                                parent.append(n_hidden[i])
                                # if idx == 0: print('flag', n[i])
                                parent_flag = False
                                break
                        if parent_flag: parent.append(hidden[0][idx].unsqueeze(0))
                        prev_idx.append(n[-1])
        else:
            max_length = self.max_out_len
            beams = [([hidden[0]], [self.root], [self.root], nodes, nodes_hiddens, hidden, tree_hidden,
                      TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], []))]
            for t in range(max_length):
                current_beams = []
                while len(beams) > 0:
                    parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append((parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b))
                        continue

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(
                        parent, left, prev, encoder_outputs, all_nums_encoder_outputs,
                        padding_hidden, seq_mask, num_mask, hidden, tree_hidden)

                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)

                    topv, topi = out_score.topk(self.beam_size)

                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        left, parent, prev = [], [], []
                        current_node_stack = self.copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = self.copy_list(b.embedding_stack)
                        current_nodes = copy.deepcopy(nodes)
                        current_nodes_hidden = self.copy_list(nodes_hiddens)
                        current_out = copy.deepcopy(b.out)
                        current_token_logit = [tl for tl in b.token_logit]

                        current_token_logit.append(token_logit)

                        out_token = int(ti)
                        current_out.append(out_token)

                        node = current_node_stack[0].pop()

                        if out_token < self.num_start:
                            current_nodes[0].append(out_token)
                            current_nodes_hidden[0].append(hidden[0])
                            generate_input = torch.LongTensor([out_token]).to(self.device)

                            left_child, right_child, node_label = self.node_generater(current_embeddings,
                                                                                      generate_input, current_context)

                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)

                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_nodes[0].pop()
                                current_nodes[0].pop()
                                current_nodes_hidden[0].pop()
                                current_nodes_hidden[0].pop()
                                current_num = sub_stree.embedding
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                            current_nodes[0].append(out_token)
                            current_nodes_hidden[0].append(hidden[0])
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)

                        parent_flag = True
                        if len(current_nodes[0]) == 0:
                            left.append(self.root)
                            prev.append(self.root)
                            parent.append(hidden[0])
                        elif current_nodes[0][-1] < self.num_start:
                            left.append(self.root)
                            prev.append(self.node_generater.embeddings(
                                torch.LongTensor([current_nodes[0][-1]]).to(self.device)))
                            parent.append(current_nodes_hidden[0][-1])
                        else:
                            left.append(current_nums_embeddings[0, current_nodes[0][-1] - self.num_start].unsqueeze(0))
                            prev.append(current_nums_embeddings[0, current_nodes[0][-1] - self.num_start].unsqueeze(0))
                            for i in range(len(current_nodes[0]) - 1, -1, -1):
                                if current_nodes[0][i] < self.num_start:
                                    parent.append(current_nodes_hidden[0][i])
                                    parent_flag = False
                                    break
                            if parent_flag: parent.append(hidden[0])
                            # parent = parent[:1]

                        current_beams.append(
                            (parent, left, prev, current_nodes, current_nodes_hidden, hidden, tree_hidden,
                             TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                      current_left_childs, current_out, current_token_logit)))
                # print('current_nodes', current_nodes)
                # print('left, parent, prev', len(parent), parent[0].size(), len(left), left[0].size(), len(prev), prev[0].size())

                beams = sorted(current_beams, key=lambda x: x[7].score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for _, _, _, _, _, _, _, b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
        token_logits = torch.stack(token_logits, dim=1)  # B x S x N
        outputs = torch.stack(outputs, dim=1)  # B x S
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                if i == -1:
                    indices.append(0)
                    masked_index.append(temp_1)
                    continue
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices).to(self.device)
        masked_index = torch.BoolTensor(masked_index).to(self.device)

        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        all_outputs = encoder_outputs.contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
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

    def copy_list(self, l):
        r = []
        if len(l) == 0:
            return r
        for i in l:
            if type(i) is list:
                r.append(self.copy_list(i))
            else:
                r.append(i)
        return r

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

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters
