# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:33:54
# @File: multiencdec.py

import copy
import random

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from mwptoolkit.module.Encoder.graph_based_encoder import GraphBasedMultiEncoder, NumEncoder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
from mwptoolkit.module.Layer.layers import TreeAttnDecoderRNN
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Layer.tree_layers import Prediction, GenerateNode, Merge
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Strategy.beam_search import TreeBeam, Beam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss, masked_cross_entropy
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask
from mwptoolkit.utils.utils import copy_list


class MultiEncDec(nn.Module):
    """
    Reference:
        Shen et al. "Solving Math Word Problems with Multi-Encoders and Multi-Decoders" in COLING 2020.
    """

    def __init__(self, config, dataset):
        super(MultiEncDec, self).__init__()
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.hop_size = config['hop_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.dropout_ratio = config['dropout_ratio']

        self.operator_nums = dataset.operator_nums
        self.generate_nums = len(dataset.generate_list)
        self.num_start1 = dataset.num_start1
        self.num_start2 = dataset.num_start2
        self.input1_size = len(dataset.in_idx2word_1)
        self.input2_size = len(dataset.in_idx2word_2)
        self.output2_size = len(dataset.out_idx2symbol_2)
        self.unk1 = dataset.out_symbol2idx_1[SpecialTokens.UNK_TOKEN]
        self.unk2 = dataset.out_symbol2idx_2[SpecialTokens.UNK_TOKEN]
        self.sos2 = dataset.out_symbol2idx_2[SpecialTokens.SOS_TOKEN]
        self.eos2 = dataset.out_symbol2idx_2[SpecialTokens.EOS_TOKEN]

        self.out_symbol2idx1 = dataset.out_symbol2idx_1
        self.out_idx2symbol1 = dataset.out_idx2symbol_1
        self.out_symbol2idx2 = dataset.out_symbol2idx_2
        self.out_idx2symbol2 = dataset.out_idx2symbol_2
        generate_list = dataset.generate_list
        self.generate_list = [self.out_symbol2idx1[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number

        try:
            self.out_sos_token1 = self.out_symbol2idx1[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token1 = None
        try:
            self.out_eos_token1 = self.out_symbol2idx1[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token1 = None
        try:
            self.out_pad_token1 = self.out_symbol2idx1[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token1 = None
        try:
            self.out_sos_token2 = self.out_symbol2idx2[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token2 = None
        try:
            self.out_eos_token2 = self.out_symbol2idx2[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token2 = None
        try:
            self.out_pad_token2 = self.out_symbol2idx2[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token2 = None
        # Initialize models
        embedder = nn.Embedding(self.input1_size, self.embedding_size)
        in_embedder = self._init_embedding_params(dataset.trainset, dataset.in_idx2word_1, config['embedding_size'],
                                                  embedder)

        self.encoder = GraphBasedMultiEncoder(input1_size=self.input1_size,
                                              input2_size=self.input2_size,
                                              embed_model=in_embedder,
                                              embedding1_size=self.embedding_size,
                                              embedding2_size=self.embedding_size // 4,
                                              hidden_size=self.hidden_size,
                                              n_layers=self.n_layers,
                                              hop_size=self.hop_size)
        self.numencoder = NumEncoder(node_dim=self.hidden_size, hop_size=self.hop_size)
        self.tree_decoder = Prediction(hidden_size=self.hidden_size, op_nums=self.operator_nums,
                                  input_size=self.generate_nums)
        self.generate = GenerateNode(hidden_size=self.hidden_size, op_nums=self.operator_nums,
                                     embedding_size=self.embedding_size)
        self.merge = Merge(hidden_size=self.hidden_size, embedding_size=self.embedding_size)
        self.attn_decoder = TreeAttnDecoderRNN(self.hidden_size, self.embedding_size, self.output2_size, self.output2_size,
                                          self.n_layers, self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()

    def forward(self, input1, input2, input_length, num_size, num_pos, num_order, parse_graph, num_stack, target1=None,
                target2=None, output_all_layers=False):
        """

        :param torch.Tensor input1:
        :param torch.Tensor input2:
        :param torch.Tensor input_length:
        :param list num_size:
        :param list num_pos:
        :param list num_order:
        :param torch.Tensor parse_graph:
        :param list num_stack:
        :param torch.Tensor | None target1:
        :param torch.Tensor | None target2:
        :param bool output_all_layers:
        :return:
        """
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])

        num_mask = []
        max_num_size = max(num_size) + len(self.generate_list)
        for i in num_size:
            d = i + len(self.generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))

        num_pos_pad = []
        max_num_pos_size = max(num_size)
        for i in range(len(num_pos)):
            temp = num_pos[i] + [-1] * (max_num_pos_size - len(num_pos[i]))
            num_pos_pad.append(temp)

        num_order_pad = []
        max_num_order_size = max(num_size)
        for i in range(len(num_order)):
            temp = num_order[i] + [0] * (max_num_order_size - len(num_order[i]))
            num_order_pad.append(temp)

        seq_mask = torch.ByteTensor(seq_mask)
        num_mask = torch.ByteTensor(num_mask)
        num_pos_pad = torch.LongTensor(num_pos_pad)
        num_order_pad = torch.LongTensor(num_order_pad)
        encoder_outputs, num_outputs, encoder_hidden, problem_output, encoder_layer_outputs = self.encoder_forward(
            input1, input2, input_length, parse_graph, num_pos, num_pos_pad, num_order_pad, output_all_layers)

        attn_decoder_hidden = encoder_hidden[:self.n_layers]  # Use last (forward) hidden state from encoder

        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output,
                                                                                   attn_decoder_hidden, num_outputs,
                                                                                   seq_mask, num_mask, num_stack,
                                                                                   target1, target2, output_all_layers)

        model_layer_outputs = {}
        if output_all_layers:
            model_layer_outputs.update(encoder_layer_outputs)
            model_layer_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_layer_outputs

    def calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'input1', 'input2', 'output1', 'output2',
        'input1 len', 'parse graph', 'num stack', 'output1 len', 'output2 len',
        'num size', 'num pos', 'num order'
        """
        input1_var = torch.tensor(batch_data['input1']).to(self.device)
        input2_var = torch.tensor(batch_data['input2']).to(self.device)
        target1 = torch.tensor(batch_data['output1']).to(self.device)
        target2 = torch.tensor(batch_data['output2']).to(self.device)
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph']).to(self.device)

        num_stack_batch = copy.deepcopy(batch_data['num stack'])
        target1_length = torch.LongTensor(batch_data['output1 len']).to(self.device)
        target2_length = torch.LongTensor(batch_data['output2 len']).to(self.device)
        num_size_batch = batch_data['num size']

        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']

        token_logits, _, all_layer_outputs = self.forward(input1_var, input2_var, input_length, num_size_batch,
                                                          num_pos_batch, num_order_batch, parse_graph, num_stack_batch,
                                                          target1, target2, output_all_layers=True)
        target1 = all_layer_outputs['target1']
        target2 = all_layer_outputs['target2']
        (tree_token_logits,attn_token_logits) = token_logits

        loss1 = masked_cross_entropy(tree_token_logits, target1, target1_length)
        loss2 = masked_cross_entropy(attn_token_logits.contiguous(),target2.contiguous(),target2_length)
        loss = loss1+loss2
        loss.backward()

        if self.USE_CUDA:
            torch.cuda.empty_cache()
        return loss

    def model_test(self, batch_data: dict) -> Tuple[str, list, list]:
        """Model test.

        :param batch_data: one batch data.
        :return: result_type, predicted equation, target equation.

        batch_data should include keywords 'input1', 'input2', 'output1', 'output2',
        'input1 len', 'parse graph', 'num stack', 'num pos', 'num order', 'num list'
        """
        input1_var = torch.tensor(batch_data['input1']).to(self.device)
        input2_var = torch.tensor(batch_data['input2']).to(self.device)
        target1 = torch.tensor(batch_data['output1']).to(self.device)
        target2 = torch.tensor(batch_data['output2']).to(self.device)
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph']).to(self.device)

        num_stack_batch = copy.deepcopy(batch_data['num stack'])

        num_size_batch = batch_data['num size']
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        num_list = batch_data['num list']

        _, outputs, all_layer_outputs = self.forward(input1_var, input2_var, input_length, num_size_batch,
                                                     num_pos_batch, num_order_batch, parse_graph, num_stack_batch,
                                                     output_all_layers=True)
        (tree_outputs,attn_outputs) = outputs
        tree_score = all_layer_outputs['tree_score']
        attn_score = all_layer_outputs['attn_score']
        if tree_score < attn_score:
            output1 = self.convert_idx2symbol1(tree_outputs[0], num_list[0], copy_list(num_stack_batch[0]))
            targets1 = self.convert_idx2symbol1(target1[0], num_list[0], copy_list(num_stack_batch[0]))

            result_type = 'tree'
            if self.USE_CUDA:
                torch.cuda.empty_cache()
            return result_type, output1, targets1
        else:
            output2 = self.convert_idx2symbol2(attn_outputs, num_list, copy_list(num_stack_batch))
            targets2 = self.convert_idx2symbol2(target2, num_list, copy_list(num_stack_batch))
            result_type = 'attn'
            if self.USE_CUDA:
                torch.cuda.empty_cache()
            return result_type, output2, targets2

    def predict(self,batch_data,output_all_layers=False):
        input1_var = torch.tensor(batch_data['input1']).to(self.device)
        input2_var = torch.tensor(batch_data['input2']).to(self.device)
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph']).to(self.device)

        num_stack_batch = copy.deepcopy(batch_data['num stack'])
        num_size_batch = batch_data['num size']
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']

        token_logits, symbol_outputs, model_all_outputs = self.forward(input1_var, input2_var, input_length,
                                                                       num_size_batch,
                                                                       num_pos_batch, num_order_batch, parse_graph,
                                                                       num_stack_batch,
                                                                       output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

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
        masked_index = torch.ByteTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices.cuda()
            masked_index = masked_index.cuda()
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index

    def generate_decoder_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        # when the decoder input is copied num but the num has two pos, chose the max
        if self.USE_CUDA:
            decoder_output = decoder_output.cpu()
        target = torch.LongTensor(target)
        for i in range(target.size(0)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
        return target

    def encoder_forward(self, input1, input2, input_length, parse_graph, num_pos, num_pos_pad, num_order_pad,
                        output_all_layers=False):
        input1 = input1.transpose(0, 1)
        input2 = input2.transpose(0, 1)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = input1.size(1)

        encoder_outputs, encoder_hidden = self.encoder(input1, input2, input_length, parse_graph)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos,
                                                                                batch_size, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs,
                                                                       num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
            all_layer_outputs['inputs_representation'] = problem_output
            all_layer_outputs['number_representation'] = num_encoder_outputs
            all_layer_outputs['num_encoder_outputs'] = num_outputs
        return encoder_outputs, num_outputs,encoder_hidden, problem_output,all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, attn_decoder_hidden, all_nums_encoder_outputs,
                        seq_mask, num_mask, num_stack, target1, target2, output_all_layers):
        num_stack1 = copy.deepcopy(num_stack)
        num_stack2 = copy.deepcopy(num_stack)
        tree_token_logits, tree_outputs, tree_layer_outputs = self.tree_decoder_forward(encoder_outputs, problem_output,
                                                                                        all_nums_encoder_outputs,
                                                                                        num_stack1, seq_mask, num_mask,
                                                                                        target1, output_all_layers)
        attn_token_logits, attn_outputs, attn_layer_outputs = self.attn_decoder_forward(encoder_outputs, seq_mask,
                                                                                        attn_decoder_hidden, num_stack2,
                                                                                        target2, output_all_layers)
        all_layer_output = {}
        if output_all_layers:
            all_layer_output.update(tree_layer_outputs)
            all_layer_output.update(attn_layer_outputs)

        return (tree_token_logits,attn_token_logits),(tree_outputs,attn_outputs),all_layer_output

    def tree_decoder_forward(self,encoder_outputs,problem_output,all_nums_encoder_outputs,nums_stack,seq_mask,num_mask,target=None,output_all_layers=False):
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
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.tree_decoder(
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
                                                                    self.num_start1, self.unk1)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = self.generate(current_embeddings, generate_input,
                                                                          current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                       node_stacks, target[t].tolist(), embeddings_stacks):
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

                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.tree_decoder(
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

                        if out_token < self.num_start1:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input.cuda()
                            left_child, right_child, node_label = self.generate(current_embeddings,
                                                                                      generate_input,
                                                                                      current_context)

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
            all_layer_outputs['tree_token_logits'] = token_logits
            all_layer_outputs['tree_outputs'] = outputs
            all_layer_outputs['target1'] = target
            if target is None:
                all_layer_outputs['tree_score'] = score
        return token_logits, outputs, all_layer_outputs

    def attn_decoder_forward(self, encoder_outputs, seq_mask, decoder_hidden, num_stack, target=None, output_all_layers=False):
        # Prepare input and output variables
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.LongTensor([self.sos2] * batch_size)

        if target is not None:
            target = target.transpose(0, 1)
        max_output_length = target.size(0) if target is not None else self.max_out_len

        token_logits = torch.zeros(max_output_length, batch_size, self.attn_decoder.output_size)
        outputs = torch.zeros(max_output_length, batch_size)

        # Move new Variables to CUDA
        if self.USE_CUDA:
            token_logits = token_logits.cuda()

        if target is not None and random.random() < self.teacher_force_ratio:
            # Run through decoder one time step at a time
            for t in range(max_output_length):
                if self.USE_CUDA:
                    decoder_input = decoder_input.cuda()

                token_logit, decoder_hidden = self.attn_decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                output = torch.topk(token_logit,1,dim=-1)[1]

                token_logits[t] = token_logit
                outputs[t] = output.squeeze(-1)

                decoder_input = self.generate_decoder_input(target[t].cpu().tolist(), token_logit, num_stack,
                                                            self.num_start2, self.unk2)
                target[t] = decoder_input
        else:
            beam_list = list()
            score = torch.zeros(batch_size)
            if self.USE_CUDA:
                score = score.cuda()
            beam_list.append(Beam(score, decoder_input, decoder_hidden, token_logits,outputs))
            # Run through decoder one time step at a time
            for t in range(max_output_length):
                beam_len = len(beam_list)
                beam_scores = torch.zeros(batch_size, self.attn_decoder.output_size * beam_len)
                all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
                all_token_logits = torch.zeros(max_output_length, batch_size * beam_len, self.attn_decoder.output_size)
                all_outputs = torch.zeros(max_output_length,batch_size*beam_len)
                if self.USE_CUDA:
                    beam_scores = beam_scores.cuda()
                    all_hidden = all_hidden.cuda()
                    all_token_logits = all_token_logits.cuda()
                    all_outputs = all_outputs.cuda()

                for b_idx in range(len(beam_list)):
                    decoder_input = beam_list[b_idx].input_var
                    decoder_hidden = beam_list[b_idx].hidden

                    if self.USE_CUDA:
                        #                    rule_mask = rule_mask.cuda()
                        decoder_input = decoder_input.cuda()

                    token_logit, decoder_hidden = self.attn_decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                  seq_mask)

                    #                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                    score = F.log_softmax(token_logit, dim=1)
                    beam_score = beam_list[b_idx].score
                    beam_score = beam_score.unsqueeze(1)
                    repeat_dims = [1] * beam_score.dim()
                    repeat_dims[1] = score.size(1)
                    beam_score = beam_score.repeat(*repeat_dims)
                    score += beam_score
                    beam_scores[:, b_idx * self.attn_decoder.output_size:(b_idx + 1) * self.attn_decoder.output_size] = score
                    all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                    beam_list[b_idx].token_logits[t] = token_logit
                    all_token_logits[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                        beam_list[b_idx].token_logits
                topv, topi = beam_scores.topk(self.beam_size, dim=1)
                beam_list = list()

                for k in range(self.beam_size):
                    temp_topk = topi[:, k]
                    temp_input = temp_topk % self.attn_decoder.output_size
                    temp_input = temp_input.data
                    if self.USE_CUDA:
                        temp_input = temp_input.cpu()
                    temp_beam_pos = temp_topk / self.attn_decoder.output_size
                    temp_beam_pos = torch.floor(temp_beam_pos).long()

                    indices = torch.LongTensor(range(batch_size))
                    if self.USE_CUDA:
                        indices = indices.cuda()
                    indices += temp_beam_pos * batch_size

                    temp_hidden = all_hidden.index_select(1, indices)
                    temp_token_logits = all_token_logits.index_select(1, indices)
                    temp_output = all_outputs.index_select(1, indices)
                    temp_output[t] = temp_input

                    beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_token_logits, temp_output))
            token_logits = beam_list[0].token_logits
            outputs = beam_list[0].outputs
            score = beam_list[0].score

            if target is not None:
                for t in range(max_output_length):
                    target[t] = self.generate_decoder_input(target[t].cpu().tolist(), token_logits[t], num_stack,
                                                            self.num_start2, self.unk2)

        token_logits = token_logits.transpose(0, 1)
        outputs = outputs.transpose(0, 1)

        if target is not None:
            target = target.transpose(0, 1)

        all_layer_outputs={}
        if output_all_layers:
            all_layer_outputs['attn_token_logits'] = token_logits
            all_layer_outputs['attn_outputs'] = outputs
            all_layer_outputs['target2'] = target
            if target is None:
                all_layer_outputs['attn_score'] = score
        return token_logits,outputs,all_layer_outputs

    def _init_embedding_params(self, train_data, vocab, embedding_size, embedder):
        sentences = []
        for data in train_data:
            sentence = []
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentences.append(sentence)
        from gensim.models import word2vec
        model = word2vec.Word2Vec(sentences, vector_size=embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        for idx in range(len(vocab)):
            if idx != pad_idx:
                emb_vectors.append(np.array(model.wv[vocab[idx]]))
            else:
                emb_vectors.append(np.zeros((embedding_size)))
        emb_vectors = np.array(emb_vectors)
        embedder.weight.data.copy_(torch.from_numpy(emb_vectors))

        return embedder

    def convert_idx2symbol1(self, output, num_list, num_stack):
        # batch_size=output.size(0)
        '''batch_size=1'''
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token1, self.out_eos_token1, self.out_pad_token1]:
                break
            symbol = self.out_idx2symbol1[idx]
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

    def convert_idx2symbol2(self, output, num_list, num_stack):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token2, self.out_eos_token2, self.out_pad_token2]:
                    break
                symbol = self.out_idx2symbol2[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                elif symbol == SpecialTokens.UNK_TOKEN:
                    try:
                        pos_list = num_stack[b_i].pop()
                        c = num_list[b_i][pos_list[0]]
                        res.append(c)
                    except:
                        res.append(symbol)
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

