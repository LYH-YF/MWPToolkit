import torch
from torch import nn
import copy
import random
from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
from mwptoolkit.module.Layer.tree_layers import *
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens

class TSN(nn.Module):
    def __init__(self,config,dataset):
        super(TSN,self).__init__()

        self.t_embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.t_decoder = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.t_node_generater = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.t_merge = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.s_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.s_decoder_1 = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_1 = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_1 = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_decoder_2 = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_2 = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_2 = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.soft_target={}

    def teacher_calculate_loss(self,batch_data):
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
        all_node_outputs=self.teacher_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)
        self.loss.reset()
        self.loss.eval_batch(all_node_outputs, target, equ_mask)
        self.loss.backward()
        return self.loss.get_loss()

    def student_calculate_loss(self,batch_data):
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        target_length = batch_data["equ len"]
        equ_mask = batch_data["equ mask"]
        batch_id = batch_data["id"]
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
        seq_emb = self.s_embedder(seq)
        pade_outputs, _ = self.s_encoder(seq_emb, seq_length)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())
        UNK_TOKEN = self.unk_token
        all_node_output1=self.student1_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        all_node_output2=self.student2_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        soft_target = self.get_soft_target(batch_id)
        
        
        return

    def model_test(self,batch_data):
        return

    def teacher_net_forward(self,encoder_outputs,problem_output,target,target_length,\
                        num_pos,nums_stack,padding_hidden,seq_mask,num_mask,unk,num_start):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size)
        #print("all_nums_encoder_outputs", all_nums_encoder_outputs.size())
        left_childs = [None for _ in range(batch_size)]  #
        embeddings_stacks = [[] for _ in range(batch_size)]  #
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.t_node_generater(current_embeddings, generate_input, current_context)
            
            left_childs = []
            
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks):
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
                        current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        return all_node_outputs


    def student1_net_forward(self,encoder_outputs,problem_output,target,target_length,\
                        num_pos,nums_stack,padding_hidden,seq_mask,num_mask,unk,num_start):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size)
        #print("all_nums_encoder_outputs", all_nums_encoder_outputs.size())
        left_childs = [None for _ in range(batch_size)]  #
        embeddings_stacks = [[] for _ in range(batch_size)]  #
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.s_node_generater(current_embeddings, generate_input, current_context)
            #print("left_child", left_child.size())
            #print("right_child", right_child.size())
            #print("node_label", node_label.size())
            left_childs = []
            #print("target", target.size())
            #print("target[:,t]", target[:,t].size())
            
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks):
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
                        current_num = self.s_merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        return all_node_outputs

    def student2_net_forward(self,encoder_outputs,problem_output,target,target_length,\
                        num_pos,nums_stack,padding_hidden,seq_mask,num_mask,unk,num_start):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size)
        #print("all_nums_encoder_outputs", all_nums_encoder_outputs.size())
        left_childs = [None for _ in range(batch_size)]  #
        embeddings_stacks = [[] for _ in range(batch_size)]  #
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.s_node_generater(current_embeddings, generate_input, current_context)
            #print("left_child", left_child.size())
            #print("right_child", right_child.size())
            #print("node_label", node_label.size())
            left_childs = []
            #print("target", target.size())
            #print("target[:,t]", target[:,t].size())
            
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks):
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
                        current_num = self.s_merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        return all_node_outputs

    @torch.no_grad()
    def init_soft_target(self,batch_data):
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        target_length = batch_data["equ len"]
        ques_id = batch_data["id"]
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
        all_node_outputs=self.teacher_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        #all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)
        for id_,soft_target in zip(ques_id,all_node_outputs):
            self.soft_target[id_]=soft_target

    def get_soft_target(self,batch_id):
        return