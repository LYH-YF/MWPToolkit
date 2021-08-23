# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 05:00:12
# @File: treelstm.py

import copy

import torch
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
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
        super().__init__()
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
        #self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]

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
        self.embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, \
                               batch_first=True, dropout=self.dropout_ratio, bidirectional=True)
        self.decoder = LSTMBasedTreeDecoder(self.embedding_size, self.hidden_size * self.num_layers, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = NodeEmbeddingLayer(self.operator_nums, self.embedding_size)
        self.root = nn.Parameter(torch.randn(1, self.embedding_size))

        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, generate_nums, num_pos, \
                num_start, target=None, target_length=None, max_length=30, beam_size=5, UNK_TOKEN=None):
        # sequence mask for attention
        beam_size = self.beam_size
        seq_mask = []
        max_len = max(seq_length)
        for i in seq_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size) + len(generate_nums)
        for i in num_size:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.embedding_size)]).unsqueeze(0).to(self.device)
        seq_emb = self.embedder(seq)
        #print('seq_emb', seq_emb.size())
        pade_outputs, initial_hidden = self.encoder(seq_emb)
        problem_output = torch.cat([pade_outputs[:, -1, :self.hidden_size], pade_outputs[:, 0, self.hidden_size:]], dim=1)
        encoder_outputs = pade_outputs

        if target != None:
            all_node_outputs = self.generate_node(encoder_outputs, problem_output, target, target_length, \
                                                  num_pos, nums_stack, padding_hidden, seq_mask, num_mask, UNK_TOKEN,
                                                  num_start, initial_hidden)
            # print('all_node_outputs', all_node_outputs); exit()
        else:
            all_node_outputs = self.generate_node_(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length, initial_hidden)
            return all_node_outputs
        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)  # B x S x N
        return all_node_outputs

    def calculate_loss(self, batch_data):
        """Finish forward-propagating, calculating loss and back-propagation.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            float: loss value.
        """
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        target_length = batch_data["equ len"]
        equ_mask = batch_data["equ mask"]
        generate_nums = self.generate_nums
        num_start = self.num_start
        UNK_TOKEN = self.unk_token

        # sequence mask for attention
        seq_mask = []
        max_len = max(seq_length)
        for i in seq_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size) + len(generate_nums)
        for i in num_size:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.embedding_size)]).unsqueeze(0).to(self.device)
        seq_emb = self.embedder(seq)
        #print('seq_emb', seq_emb.size())
        pade_outputs, initial_hidden = self.encoder(seq_emb)
        problem_output = torch.cat([pade_outputs[:, -1, :self.hidden_size], pade_outputs[:, 0, self.hidden_size:]], dim=1)
        encoder_outputs = pade_outputs
        all_node_outputs = self.generate_node(encoder_outputs, problem_output, target, target_length, \
                                                  num_pos, nums_stack, padding_hidden, seq_mask, num_mask, UNK_TOKEN,
                                                  num_start, initial_hidden)
        all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)
        target_length = target_length.to(self.device)
        self.loss.reset()
        self.loss.eval_batch(all_node_outputs, target, target_length)
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data):
        """Model test.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        target_length = batch_data["equ len"]
        equ_mask = batch_data["equ mask"]
        num_list = batch_data['num list']
        generate_nums = self.generate_nums
        num_start = self.num_start
        UNK_TOKEN = self.unk_token

        # sequence mask for attention
        beam_size = self.beam_size
        max_length = self.max_out_len
        seq_mask = []
        max_len = max(seq_length)
        for i in seq_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(self.device)

        num_mask = []
        max_num_size = max(num_size) + len(generate_nums)
        for i in num_size:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask).to(self.device)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.embedding_size)]).unsqueeze(0).to(self.device)
        seq_emb = self.embedder(seq)
        #print('seq_emb', seq_emb.size())
        pade_outputs, initial_hidden = self.encoder(seq_emb)
        problem_output = torch.cat([pade_outputs[:, -1, :self.hidden_size], pade_outputs[:, 0, self.hidden_size:]], dim=1)
        encoder_outputs = pade_outputs

        all_node_outputs = self.generate_node_(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length, initial_hidden)
        all_outputs = self.convert_idx2symbol(all_node_outputs, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_outputs, targets


    def generate_node(self, encoder_outputs, problem_output, target, target_length, \
                      num_pos, nums_stack, padding_hidden, seq_mask, num_mask, unk, num_start, initial_hidden):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size * self.num_layers)
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        nodes = [[] for _ in range(batch_size)]

        hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))
        tree_hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))
        #tree_hidden = (torch.zeros(batch_size, self.hidden_size*self.num_layers).to(self.device), torch.zeros(batch_size, self.hidden_size*self.num_layers).to(self.device))
        nodes_hiddens = [[] for _ in range(batch_size)]

        parent = [hidden[0][idx].unsqueeze(0) for idx in range(batch_size)]
        left = [self.root for _ in range(batch_size)]
        prev = [self.root for _ in range(batch_size)]

        for t in range(max_target_length):
            #print('t', t, all_nums_encoder_outputs.size())
            num_score, op, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(parent, left, prev, encoder_outputs, all_nums_encoder_outputs,
                                                                                                                            padding_hidden, seq_mask, num_mask, hidden, tree_hidden)

            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            # print("left_child, right_child, node_label", left_child, right_child, node_label); exit()

            left, parent, prev, prev_idx = [], [], [], []
            left_childs = []
            for idx, l, r, node_stack, i, o, n, n_hidden in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks, nodes,
                                                                nodes_hiddens):
                continue_flag = False
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue_flag = True
                    #continue

                if not continue_flag:
                    if i < num_start:
                        n.append(i)
                        n_hidden.append(hidden[0][idx].unsqueeze(0))
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
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
                # elif False: #len(n) == 0:
                #     left.append(self.root)
                #     parent.append(self.root)
                #     prev.append(self.root)
                #     prev_idx.append(None)
                elif n[-1] < num_start:
                    left.append(self.root)
                    parent.append(n_hidden[-1])
                    prev.append(self.node_generater.embeddings(torch.LongTensor([n[-1]]).to(self.device)))
                    prev_idx.append(n[-1])
                else:
                    left.append(current_nums_embeddings[idx, n[-1] - num_start].unsqueeze(0))
                    prev.append(current_nums_embeddings[idx, n[-1] - num_start].unsqueeze(0))
                    for i in range(len(n) - 1, -1, -1):
                        if n[i] < num_start:
                            parent.append(n_hidden[i])
                            #if idx == 0: print('flag', n[i])
                            parent_flag = False
                            break
                    if parent_flag: parent.append(hidden[0][idx].unsqueeze(0))
                    prev_idx.append(n[-1])
                #if len(parent)
                #if idx == 0: print('prev_idx', prev_idx, n)

            #print('left, parent, prev', len(parent), parent[0].size(), len(left), left[0].size(), len(prev), prev[0].size())

        #exit()
        return all_node_outputs

    def generate_node_(self, encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, \
                       num_start, beam_size, max_length, initial_hidden):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        nodes = [[] for _ in range(batch_size)]
        nodes_hiddens = [[] for _ in range(batch_size)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size * self.num_layers)

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        #hidden = (initial_hidden[0].squeeze(0), initial_hidden[1].squeeze(0)) #initial_hidden #(torch.zeros(batch_size, self.hidden_size).to(self.device), torch.zeros(batch_size, self.hidden_size).to(self.device))
        hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))
        #tree_hidden = (torch.zeros(batch_size, self.hidden_size*self.num_layers).to(self.device), torch.zeros(batch_size, self.hidden_size*self.num_layers).to(self.device))
        tree_hidden = (initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1))

        beams = [([hidden[0]], [self.root], [self.root], nodes, nodes_hiddens, hidden, tree_hidden, TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, []))]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append((parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b))
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(parent, left, prev, encoder_outputs, all_nums_encoder_outputs,
                                                                                                                                padding_hidden, seq_mask, num_mask, hidden, tree_hidden)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                #print('out_score', out_score.size())

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    left, parent, prev = [], [], []
                    current_node_stack = self.copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = self.copy_list(b.embedding_stack)
                    current_nodes = copy.deepcopy(nodes)
                    current_nodes_hidden = self.copy_list(nodes_hiddens)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        current_nodes[0].append(out_token)
                        current_nodes_hidden[0].append(hidden[0])
                        generate_input = torch.LongTensor([out_token]).to(self.device)

                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

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

                    #if idx == 0: print('n', i, n, node_stack)
                    parent_flag = True
                    if len(current_nodes[0]) == 0:
                        left.append(self.root)
                        prev.append(self.root)
                        parent.append(hidden[0])
                    elif current_nodes[0][-1] < num_start:
                        left.append(self.root)
                        prev.append(self.node_generater.embeddings(torch.LongTensor([current_nodes[0][-1]]).to(self.device)))
                        parent.append(current_nodes_hidden[0][-1])
                    else:
                        left.append(current_nums_embeddings[0, current_nodes[0][-1] - num_start].unsqueeze(0))
                        prev.append(current_nums_embeddings[0, current_nodes[0][-1] - num_start].unsqueeze(0))
                        for i in range(len(current_nodes[0]) - 1, -1, -1):
                            if current_nodes[0][i] < num_start:
                                parent.append(current_nodes_hidden[0][i])
                                parent_flag = False
                                break
                        if parent_flag: parent.append(hidden[0])
                        #parent = parent[:1]

                    current_beams.append((parent, left, prev, current_nodes, current_nodes_hidden, hidden, tree_hidden,
                                          TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out)))
            #print('current_nodes', current_nodes)
            #print('left, parent, prev', len(parent), parent[0].size(), len(left), left[0].size(), len(prev), prev[0].size())

            beams = sorted(current_beams, key=lambda x: x[7].score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for _, _, _, _, _, _, _, b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        # print('beams[0][4].out', beams[0][4].out)
        # exit()
        return beams[0][7].out

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

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters
