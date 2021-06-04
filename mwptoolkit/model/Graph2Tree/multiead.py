import copy
import random

import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Encoder.graph_based_encoder import GraphBasedMultiEncoder, NumEncoder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
from mwptoolkit.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Strategy.beam_search import TreeBeam, Beam
from mwptoolkit.utils.enum_type import SpecialTokens
from mwptoolkit.utils.utils import copy_list


class MultiEAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input1_size = config['input1_size']
        self.input2_size = config['input2_size']
        self.output2_size = config['output2_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.hop_size = config['hop_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.oprator_nums = config["operator_nums"]
        self.generate_nums = config["generate_size"]
        self.dropout_ratio = config['dropout_ratio']
        self.num_start1 = config['num_start1']
        self.num_start2 = config['num_start2']
        self.unk1 = config['out_symbol2idx1'][SpecialTokens.UNK_TOKEN]
        self.unk2 = config['out_symbol2idx2'][SpecialTokens.UNK_TOKEN]
        self.sos = config['out_symbol2idx2'][SpecialTokens.SOS_TOKEN]
        self.eos2 = config['out_symbol2idx2'][SpecialTokens.EOS_TOKEN]

        # Initialize models
        self.embedder = BaiscEmbedder(self.input1_size, self.embedding_size)

        self.encoder = GraphBasedMultiEncoder(input1_size=self.input1_size,
                                              input2_size=self.input2_size,
                                              embed_model=self.embedder,
                                              embedding1_size=self.embedding_size,
                                              embedding2_size=self.embedding_size // 4,
                                              hidden_size=self.hidden_size,
                                              n_layers=self.n_layers,
                                              hop_size=self.hop_size)

        self.numencoder = NumEncoder(node_dim=self.hidden_size, hop_size=self.hop_size)

        self.predict = TreeDecoder(hidden_size=self.hidden_size, op_nums=self.oprator_nums, generate_size=self.generate_nums)

        self.generate = NodeGenerater(hidden_size=self.hidden_size, op_nums=self.oprator_nums, embedding_size=self.embedding_size)

        self.merge = SubTreeMerger(hidden_size=self.hidden_size, embedding_size=self.embedding_size)

        self.decoder = AttentionalRNNDecoder(embedding_size=self.embedding_size,
                                             hidden_size=self.hidden_size,
                                             context_size=self.hidden_size,
                                             num_dec_layers=self.n_layers,
                                             rnn_cell_type='gru',
                                             dropout_ratio=self.dropout_ratio)
        self.out = nn.Linear(self.hidden_size,self.output2_size)

    def forward(self,input1_var, input2_var, input_length, target1, target1_length, target2, target2_length,\
                num_stack_batch, num_size_batch,generate_list,num_pos_batch, num_order_batch, parse_graph):
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_list)
        for i in num_size_batch:
            d = i + len(generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask)

        num_pos_pad = []
        max_num_pos_size = max(num_size_batch)
        for i in range(len(num_pos_batch)):
            temp = num_pos_batch[i] + [-1] * (max_num_pos_size - len(num_pos_batch[i]))
            num_pos_pad.append(temp)
        num_pos_pad = torch.LongTensor(num_pos_pad)

        num_order_pad = []
        max_num_order_size = max(num_size_batch)
        for i in range(len(num_order_batch)):
            temp = num_order_batch[i] + [0] * (max_num_order_size - len(num_order_batch[i]))
            num_order_pad.append(temp)
        num_order_pad = torch.LongTensor(num_order_pad)

        num_stack1_batch = copy.deepcopy(num_stack_batch)
        num_stack2_batch = copy.deepcopy(num_stack_batch)
        #num_start2 = output2_lang.n_words - copy_nums - 2
        #unk1 = output1_lang.word2index["UNK"]
        #unk2 = output2_lang.word2index["UNK"]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        # input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
        # input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
        # target1 = torch.LongTensor(target1_batch).transpose(0, 1)
        # target2 = torch.LongTensor(target2_batch).transpose(0, 1)
        input1_var = input1_var.transpose(0, 1)
        input2_var = input2_var.transpose(0, 1)
        target1 = target1.transpose(0, 1)
        target2 = target2.transpose(0, 1)
        parse_graph_pad = torch.LongTensor(parse_graph)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)

        encoder_outputs, encoder_hidden = self.encoder(input1_var, input2_var, input_length, parse_graph_pad)
        copy_num_len = [len(_) for _ in num_pos_batch]
        num_size = max(copy_num_len)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

        decoder_hidden = encoder_hidden[self.n_layers]  # Use last (forward) hidden state from encoder
        if target1 != None:
            all_output1 = self.train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length,
                                                batch_size, padding_hidden, seq_mask,num_mask, num_pos_batch, 
                                                num_order_pad, num_stack1_batch)
    
            all_output2 = self.train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length,
                                            batch_size, seq_mask, num_stack2_batch)
            return "train",all_output1,all_output2
        else:
            all_output1=self.evaluate_tree_double(encoder_outputs, problem_output, num_outputs,batch_size, padding_hidden, seq_mask,num_mask)
            all_output2=self.evaluate_attn_double(encoder_outputs,decoder_hidden,batch_size,seq_mask)
            if all_output1.score >= all_output2.score:
                return "tree", all_output1.out, all_output1.score
            else:
                return "attn", all_output2.all_output, all_output2.score

    def train_tree_double(self, encoder_outputs, problem_output, all_nums_encoder_outputs, target, target_length, batch_size, padding_hidden, seq_mask, num_mask, num_pos, num_order_pad,
                          nums_stack_batch):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []

        #    copy_num_len = [len(_) for _ in num_pos]
        #    num_size = max(copy_num_len)
        #    nums_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
        #                                                                        encoder.hidden_size)
        #    all_nums_encoder_outputs = numencoder(nums_encoder_outputs, num_order_pad)
        #    all_nums_encoder_outputs = all_nums_encoder_outputs.masked_fill_(masked_index.bool(), 0.0)

        # num_start = output_lang.num_start
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch)
            target[t] = target_t
            # if USE_CUDA:
            #     generate_input = generate_input.cuda()
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < self.num_start1:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - self.num_start1].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        # target = target.transpose(0, 1).contiguous()
        # if USE_CUDA:
        #     # all_leafs = all_leafs.cuda()
        #     all_node_outputs = all_node_outputs.cuda()
        #     target = target.cuda()

        # # op_target = target < num_start
        # # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        # loss = masked_cross_entropy(all_node_outputs, target, target_length)
        # # loss = loss_0 + loss_1
        return all_node_outputs  # , loss_0.item(), loss_1.item()

    def train_attn_double(self, encoder_outputs, decoder_hidden, target, target_length, batch_size, seq_mask, nums_stack_batch):
        # Prepare input and output variables
        decoder_input = torch.LongTensor([self.sos] * batch_size).to(self.device)

        max_target_length = max(target_length)
        all_decoder_outputs = torch.zeros(max_target_length, batch_size, self.output_size).to(self.device)

        # Move new Variables to CUDA
        # if USE_CUDA:
        #     all_decoder_outputs = all_decoder_outputs.cuda()

        if random.random() < self.teacher_force_ratio:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                # if USE_CUDA:
                #     decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                all_decoder_outputs[t] = decoder_output
                decoder_output = self.out(decoder_output)
                decoder_input = self.generate_decoder_input(target[t], decoder_output, nums_stack_batch)
                target[t] = decoder_input
        else:
            beam_list = list()
            score = torch.zeros(batch_size).to(self.device)
            # if USE_CUDA:
            #     score = score.cuda()
            beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                beam_len = len(beam_list)
                beam_scores = torch.zeros(batch_size, self.output2_size * beam_len).to(self.device)
                all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2)).to(self.device)
                all_outputs = torch.zeros(max_target_length, batch_size * beam_len, self.output2_size).to(self.device)
                # if USE_CUDA:
                #     beam_scores = beam_scores.cuda()
                #     all_hidden = all_hidden.cuda()
                #     all_outputs = all_outputs.cuda()

                for b_idx in range(len(beam_list)):
                    decoder_input = beam_list[b_idx].input_var
                    decoder_hidden = beam_list[b_idx].hidden

                    #                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                    #                                               num_start, copy_nums, generate_nums, english)
                    # if USE_CUDA:
                    #                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.to(self.device)

                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                    decoder_output = self.out(decoder_output)
                    #                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                    score = F.log_softmax(decoder_output, dim=1)
                    beam_score = beam_list[b_idx].score
                    beam_score = beam_score.unsqueeze(1)
                    repeat_dims = [1] * beam_score.dim()
                    repeat_dims[1] = score.size(1)
                    beam_score = beam_score.repeat(*repeat_dims)
                    score += beam_score
                    beam_scores[:, b_idx * self.output2_size:(b_idx + 1) * self.output2_size] = score
                    all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                    beam_list[b_idx].all_output[t] = decoder_output
                    all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                        beam_list[b_idx].all_output
                topv, topi = beam_scores.topk(self.beam_size, dim=1)
                beam_list = list()

                for k in range(self.beam_size):
                    temp_topk = topi[:, k]
                    temp_input = temp_topk % self.output2_size
                    temp_input = temp_input.data
                    # if USE_CUDA:
                    #     temp_input = temp_input.cpu()
                    temp_beam_pos = temp_topk / self.decoder.output2_size

                    indices = torch.LongTensor(range(batch_size)).to(self.device)
                    # if USE_CUDA:
                    #     indices = indices.cuda()
                    indices += temp_beam_pos * batch_size

                    temp_hidden = all_hidden.index_select(1, indices)
                    temp_output = all_outputs.index_select(1, indices)

                    beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
            all_decoder_outputs = beam_list[0].all_output

            for t in range(max_target_length):
                target[t] = self.generate_decoder_input(target[t], all_decoder_outputs[t], nums_stack_batch)
        # Loss calculation and backpropagation

        # if USE_CUDA:
        #     target = target.cuda()

        # loss = masked_cross_entropy(
        #     all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        #     target.transpose(0, 1).contiguous(),  # -> batch x seq
        #     target_length
        # )

        return all_decoder_outputs

    def evaluate_tree_double(self,encoder_outputs, problem_output, all_nums_encoder_outputs, 
                          batch_size, padding_hidden, seq_mask, num_mask):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        #num_start = output_lang.num_start
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(self.max_out_len):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                topv, topi = out_score.topk(self.beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < self.num_start1:
                        generate_input = torch.LongTensor([out_token]).to(self.device)
                        # if USE_CUDA:
                        #     generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - self.num_start1].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:self.beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0]

    def evaluate_attn_double(self,encoder_outputs, decoder_hidden, batch_size, seq_mask):
        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([self.sos])  # SOS
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(self.max_out_len):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == self.eos2:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0]
            beam_scores = torch.zeros(self.output2_size * beam_len).to(self.device)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2).to(self.device)
            # if USE_CUDA:
            #     beam_scores = beam_scores.cuda()
            #     all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == self.eos2:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
                #                                1, num_start, copy_nums, generate_nums, english)
                #if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.to(self.device)

                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = F.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * self.output2_size: (current_idx + 1) * self.output2_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(self.beam_size)

            for k in range(self.beam_size):
                word_n = int(topi[k])
                word_input = word_n % self.output2_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / self.output2_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < self.beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:self.beam_size]
        return beam_list[0]

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        # indices = torch.LongTensor(indices)
        # masked_index = torch.ByteTensor(masked_index)
        indices = torch.LongTensor(indices).to(self.device)
        masked_index = torch.BoolTensor(masked_index).to(self.device)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index

    def generate_tree_input(self, target, decoder_output, nums_stack_batch):
        # when the decoder input is copied num but the num has two pos, chose the max
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == self.unk1:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, self.num_start1 + num] > max_score:
                        target[i] = num + self.num_start1
                        max_score = decoder_output[i, self.num_start1 + num]
            if target_input[i] >= self.num_start1:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def generate_decoder_input(self, target, decoder_output, nums_stack_batch):
        # when the decoder input is copied num but the num has two pos, chose the max
        # if USE_CUDA:
        #     decoder_output = decoder_output.cpu()
        for i in range(target.size(0)):
            if target[i] == self.unk2:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, self.num_start2 + num] > max_score:
                        target[i] = num + self.num_start2
                        max_score = decoder_output[i, self.num_start2 + num]
        return target

    # def get_all_number_encoder_outputs(self,encoder_outputs, num_pos, num_size, hidden_size):
    #     indices = list()
    #     sen_len = encoder_outputs.size(1)
    #     batch_size=encoder_outputs.size(0)
    #     masked_index = []
    #     temp_1 = [1 for _ in range(hidden_size)]
    #     temp_0 = [0 for _ in range(hidden_size)]
    #     for b in range(batch_size):
    #         for i in num_pos[b]:
    #             indices.append(i + b * sen_len)
    #             masked_index.append(temp_0)
    #         indices += [0 for _ in range(len(num_pos[b]), num_size)]
    #         masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    #     indices = torch.LongTensor(indices).to(self.device)
    #     masked_index = torch.BoolTensor(masked_index).to(self.device)

    #     masked_index = masked_index.view(batch_size, num_size, hidden_size)
    #     all_outputs = encoder_outputs.contiguous()
    #     all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    #     all_num = all_embedding.index_select(0, indices)
    #     all_num = all_num.view(batch_size, num_size, hidden_size)
    #     return all_num.masked_fill_(masked_index, 0.0)