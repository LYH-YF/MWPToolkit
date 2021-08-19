# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/19 11:29:13
# @File: sausolver.py


import random
import torch
from torch import nn
import copy

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Decoder.tree_decoder import SARTreeDecoder
from mwptoolkit.module.Layer.tree_layers import NodeGenerater, SubTreeMerger, TreeNode, TreeEmbedding
from mwptoolkit.module.Layer.tree_layers import Prediction,GenerateNode,Merge
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss,masked_cross_entropy
from mwptoolkit.loss.mse_loss import MSELoss
from mwptoolkit.utils.utils import copy_list
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class SAUSolver(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        #parameter
        self.hidden_size = config["hidden_size"]
        self.device = config["device"]
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.loss_weight=config['loss_weight']

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
        #module
        self.embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        #self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size,self.hidden_size,self.num_layers,self.rnn_cell_type,self.dropout_ratio,batch_first=False)
        self.decoder = SARTreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        
        self.loss1 = MaskedCrossEntropyLoss()
        #
    def calculate_loss(self,batch_data):
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
        # sequence mask for attention
        unk=self.unk_token
        
        '''
        input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               embedder,encoder, predict, generate, merge, num_pos,batch_graph,unk,num_start
        '''
        loss = self.train_tree(seq,seq_length,target,target_length,\
            nums_stack,num_size,generate_nums,num_pos,unk,num_start)
        return loss
    
    def model_test(self,batch_data):
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
        # sequence mask for attention
        all_node_output = self.evaluate_tree(seq,seq_length,generate_nums,num_pos,num_start,self.beam_size,self.max_out_len)
        
        all_output = self.convert_idx2symbol(all_node_output, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output,targets
    
    def train_tree(self,input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                    num_pos,unk,num_start, english=False):
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_nums)
        for i in num_size_batch:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0, 1)

        target = target_batch.transpose(0, 1)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)

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

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        sub_tree_outputs = []
        sub_tree_target = []
        sub_tree_mask = []

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                self.hidden_size)

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if self.USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            left_childs = []
            sub_tree_emb=[]
            loss_mask = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                    sub_tree_emb.append(o[-1].embedding)
                    loss_mask.append(torch.ones(1,dtype=torch.float))
                else:
                    left_childs.append(None)
                    sub_tree_emb.append(padding_hidden)
                    loss_mask.append(torch.zeros(1,dtype=torch.float))
            
            sub_tree_emb = torch.stack(sub_tree_emb)
            loss_mask = torch.stack(loss_mask)
            score = self.decoder.attn(sub_tree_emb, encoder_outputs, seq_mask)
            s_aligned_vector = score.bmm(encoder_outputs)
            s_aligned_a, s_aligned_d=self.decoder.Semantically_Aligned_Regularization(sub_tree_emb,s_aligned_vector)
            sub_tree_outputs.append(s_aligned_a)
            sub_tree_target.append(s_aligned_d)
            sub_tree_mask.append(loss_mask)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)# B x S x N
        sub_tree_outputs = torch.cat(sub_tree_outputs,dim=1)
        sub_tree_target = torch.cat(sub_tree_target,dim=1)
        sub_tree_mask = torch.cat(sub_tree_mask,dim=1)

        target = target.transpose(0, 1).contiguous()
        if self.USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            target_length = torch.LongTensor(target_length).cuda()
        else:
            target_length = torch.LongTensor(target_length)
        
        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss_1 = masked_cross_entropy(all_node_outputs, target, target_length)
        loss_2=self.mse_loss(sub_tree_outputs, sub_tree_target,sub_tree_mask)
        #self.loss2.eval_batch(sub_tree_outputs, sub_tree_target)
        loss=loss_1+self.loss_weight*loss_2
        loss.backward()
        return loss.item()
        

    def evaluate_tree(self,input_batch, input_length, generate_nums, num_pos,
                  num_start,beam_size=5, max_length=30):

        seq_mask = torch.BoolTensor(1, input_length).fill_(0)
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0,1)

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

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                
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
                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)

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
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
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

    def get_all_number_encoder_outputs(self,encoder_outputs, num_pos, batch_size, num_size, hidden_size):
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

    def mse_loss(self,outputs,targets,mask=None):
        # outputs   : [batch_size,output_len,hidden_size]
        # targets   : [batch_size,output_len,hidden_size]
        # mask      : [batch_size,output_len]
        mask = mask.to(self.device)
        x = torch.sqrt(torch.sum(torch.square((outputs-targets)),dim=-1)) # [batch_size,output_len]
        y = torch.sum(x*mask,dim=-1) / torch.sum(mask,dim=-1)   # [batch_size]
        return torch.sum(y)

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



class SAUSolver_(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        #parameter
        self.hidden_size = config["hidden_size"]
        self.device = config["device"]
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.loss_weight=config['loss_weight']

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
        #module
        self.embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.decoder = SARTreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.loss1 = MaskedCrossEntropyLoss()
        #
    def forward(self,seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                num_start,target=None, target_length=None,max_length=30,beam_size=5,UNK_TOKEN=None):
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

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.embedder(seq)
        pade_outputs, _ = self.encoder(seq_emb, seq_length)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())

        if target != None:
            all_node_outputs, target=self.generate_node(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        else:
            all_node_outputs = self.generate_node_(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length)
            return all_node_outputs
        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)  # B x S x N
        return all_node_outputs, target

    def calculate_loss(self, batch_data):
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

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.embedder(seq)
        pade_outputs, _ = self.encoder(seq_emb, seq_length)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())
        UNK_TOKEN = self.unk_token

        all_node_outputs, sub_tree_outputs,sub_tree_target,sub_tree_mask=self.generate_node(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)
        sub_tree_outputs = torch.cat(sub_tree_outputs,dim=1)
        sub_tree_target = torch.cat(sub_tree_target,dim=1)
        sub_tree_mask = torch.cat(sub_tree_mask,dim=1)
        #sub_tree_outputs = sub_tree_outputs.view(-1,sub_tree_outputs.size(-1))
        #sub_tree_target = sub_tree_target.view(-1,sub_tree_target.size(-1))
        self.loss1.reset()
        #self.loss2.reset()
        self.loss1.eval_batch(all_node_outputs, target, equ_mask)
        loss_2=self.mse_loss(sub_tree_outputs, sub_tree_target,sub_tree_mask)
        #self.loss2.eval_batch(sub_tree_outputs, sub_tree_target)
        loss=self.loss1.acc_loss+self.loss_weight*loss_2
        loss.backward()
        return loss.item()

    def model_test(self, batch_data):
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        num_list = batch_data['num list']
        #target_length=batch_data["equ len"]
        generate_nums = self.generate_nums
        num_start = self.num_start

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

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.embedder(seq)
        pade_outputs, _ = self.encoder(seq_emb, seq_length)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())

        all_node_outputs = self.generate_node_(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length)
        all_outputs = self.convert_idx2symbol(all_node_outputs, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_outputs, targets

    def generate_node(self,encoder_outputs,problem_output,target,target_length,\
                        num_pos,nums_stack,padding_hidden,seq_mask,num_mask,unk,num_start):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        sub_tree_outputs = []
        sub_tree_target = []
        sub_tree_mask = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size)
        #print("all_nums_encoder_outputs", all_nums_encoder_outputs.size())
        left_childs = [None for _ in range(batch_size)]  #
        embeddings_stacks = [[] for _ in range(batch_size)]  #
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = \
                self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, \
                    padding_hidden, seq_mask, num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            #print("left_child", left_child.size())
            #print("right_child", right_child.size())
            #print("node_label", node_label.size())
            left_childs = []
            #print("target", target.size())
            #print("target[:,t]", target[:,t].size())
            sub_tree_emb=[]
            loss_mask = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    sub_tree_emb.append(padding_hidden)
                    loss_mask.append(torch.zeros(1,dtype=torch.float))
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    try:
                        current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    except:
                            print('current_num_emb:',current_nums_embeddings.size(),'num start:',self.num_start,'token idx:',i)
                            print('out list:',self.out_idx2symbol,'out gen:',self.generate_size)
                            print('batch i:',i)
                            raise ValueError
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                    sub_tree_emb.append(o[-1].embedding)
                    loss_mask.append(torch.ones(1,dtype=torch.float))
                else:
                    left_childs.append(None)
                    sub_tree_emb.append(padding_hidden)
                    loss_mask.append(torch.zeros(1,dtype=torch.float))
            
            # 
            sub_tree_emb = torch.stack(sub_tree_emb)
            loss_mask = torch.stack(loss_mask)
            score = self.decoder.attn(sub_tree_emb, encoder_outputs, seq_mask)
            s_aligned_vector = score.bmm(encoder_outputs)
            s_aligned_a, s_aligned_d=self.decoder.Semantically_Aligned_Regularization(sub_tree_emb,s_aligned_vector)
            sub_tree_outputs.append(s_aligned_a)
            sub_tree_target.append(s_aligned_d)
            sub_tree_mask.append(loss_mask)
        return all_node_outputs, sub_tree_outputs, sub_tree_target, sub_tree_mask

    def generate_node_(self,encoder_outputs,problem_output,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,beam_size,max_length):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.encoder.hidden_size)

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

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                                                                                                           seq_mask, num_mask)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = self.copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = self.copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token]).to(self.device)

                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        try:
                            current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                        except:
                            print('current_num_emb:',current_nums_embeddings.size(),'num start:',self.num_start,'token idx:',out_token)
                            print('operator:',op.size(),'num:',num_score.size(),'out:',out_score.size())
                            raise ValueError
                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        return beams[0].out

    def mse_loss(self,outputs,targets,mask=None):
        # outputs   : [batch_size,output_len,hidden_size]
        # targets   : [batch_size,output_len,hidden_size]
        # mask      : [batch_size,output_len]
        mask = mask.to(self.device)
        x = torch.sqrt(torch.sum(torch.square((outputs-targets)),dim=-1)) # [batch_size,output_len]
        y = torch.sum(x*mask,dim=-1) / torch.sum(mask,dim=-1)   # [batch_size]
        return torch.sum(y)

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
            '''
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            '''
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def generate_tree_input_(self, target, decoder_output, nums_stack_batch, num_start, unk):
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


