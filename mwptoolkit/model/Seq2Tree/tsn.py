import torch
from torch import nn
import copy
import itertools
import random
from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Encoder.graph_based_encoder import GraphBasedEncoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Decoder.tree_decoder import TreeDecoder
from mwptoolkit.module.Layer.tree_layers import *
from mwptoolkit.module.Strategy.beam_search import TreeBeam
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss,masked_cross_entropy
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens
from mwptoolkit.utils.utils import copy_list,str2float

class TSN(nn.Module):
    def __init__(self,config,dataset):
        super(TSN,self).__init__()
        #parameter
        self.hidden_size = config["hidden_size"]
        self.bidirectional = config["bidirectional"]
        self.device = config["device"]
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.alpha = 0.15

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

        self.t_embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        #self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.t_encoder = GraphBasedEncoder(self.embedding_size,self.hidden_size,self.rnn_cell_type,\
                                        self.bidirectional,self.num_layers,self.dropout_ratio)
        self.t_decoder = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.t_node_generater = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.t_merge = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        #self.s_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio)
        self.s_encoder = GraphBasedEncoder(self.embedding_size,self.hidden_size,self.rnn_cell_type,\
                                        self.bidirectional,self.num_layers,self.dropout_ratio)
        self.s_decoder_1 = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_1 = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_1 = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.s_decoder_2 = TreeDecoder(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_2 = NodeGenerater(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_2 = SubTreeMerger(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()
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
        group_nums = batch_data['group nums']
        num_list = batch_data['num list']
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

        graphs = self.build_graph(seq_length, num_list, num_pos, group_nums)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.t_embedder(seq)
        pade_outputs, _ = self.t_encoder(seq_emb, seq_length, graphs)
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
        group_nums = batch_data['group nums']
        num_list = batch_data['num list']
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

        graphs = self.build_graph(seq_length, num_list, num_pos, group_nums)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.s_embedder(seq)
        pade_outputs, _ = self.s_encoder(seq_emb, seq_length, graphs)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())
        encoder_outputs_mask = self.encoder_mask[:encoder_outputs.size(0),:max(seq_length),:].float()
        encoder_outputs_ = encoder_outputs*encoder_outputs_mask.float()

        UNK_TOKEN = self.unk_token
        all_node_output1=self.student1_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        all_node_output2=self.student2_net_forward(encoder_outputs_,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        all_node_output1 = torch.stack(all_node_output1, dim=1).to(self.device)
        all_node_output2 = torch.stack(all_node_output2, dim=1).to(self.device)
        soft_target = self.get_soft_target(batch_id)
        soft_target = torch.cat(soft_target,dim=0).to(self.device)
        
        loss1 = masked_cross_entropy(all_node_output1, target, equ_mask)
        loss2 = self.soft_target_loss(all_node_output1,soft_target,target_length)
        loss3 = masked_cross_entropy(all_node_output2, target, equ_mask)
        loss4 = self.soft_target_loss(all_node_output2,soft_target,target_length)
        cos_loss = self.cosine_loss(all_node_output1,all_node_output2,target_length)
        loss = (1-self.alpha)*loss1 + self.alpha*loss2 + (1-self.alpha)*loss3 + self.alpha*loss4 + 0.1*cos_loss
        loss.backward()
        return loss.item()

    def teacher_test(self,batch_data):
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        num_list = batch_data['num list']
        group_nums = batch_data['group nums']
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

        graphs = self.build_graph(seq_length, num_list, num_pos, group_nums)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.t_embedder(seq)
        pade_outputs, _ = self.t_encoder(seq_emb, seq_length, graphs)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())

        all_node_outputs = self.teacher_test_forward(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length)
        all_outputs = self.convert_idx2symbol(all_node_outputs, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_outputs, targets
    
    def student_test(self,batch_data):
        seq = batch_data["question"]
        seq_length = batch_data["ques len"]
        nums_stack = batch_data["num stack"]
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        num_list = batch_data['num list']
        group_nums = batch_data['group nums']
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

        graphs = self.build_graph(seq_length, num_list, num_pos, group_nums)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.s_embedder(seq)
        pade_outputs, _ = self.s_encoder(seq_emb, seq_length, graphs)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]

        all_node_output1,score1 = self.student1_test_forward(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length)
        all_node_output2,score2 = self.student2_test_forward(encoder_outputs, problem_output, padding_hidden, seq_mask, num_mask, num_pos, num_start, beam_size, max_length)
        all_output1 = self.convert_idx2symbol(all_node_output1, num_list[0], copy_list(nums_stack[0]))
        all_output2 = self.convert_idx2symbol(all_node_output2, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))

        return all_output1,score1,all_output2,score2,targets

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
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)
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
                        current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
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
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_2(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input_(target[:, t].tolist(), outputs, nums_stack, num_start, unk)
            target[:, t] = target_t
            generate_input = generate_input.to(self.device)
            left_child, right_child, node_label = self.s_node_generater_2(current_embeddings, generate_input, current_context)
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
                        current_num = self.s_merge_2(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        return all_node_outputs

    def teacher_test_forward(self,encoder_outputs,problem_output,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,beam_size,max_length):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.t_encoder.hidden_size)

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

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
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
                        generate_input = torch.LongTensor([out_token]).to(self.device)

                        left_child, right_child, node_label = self.t_node_generater(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
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
    
    def student1_test_forward(self,encoder_outputs,problem_output,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,beam_size,max_length):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        #num_size = len(num_pos[0])
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.s_encoder.hidden_size)

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

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
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
                        generate_input = torch.LongTensor([out_token]).to(self.device)

                        left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
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
        return beams[0].out,beams[0].score
    def student2_test_forward(self,encoder_outputs,problem_output,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,beam_size,max_length):
        batch_size = encoder_outputs.size(0)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        #num_size = len(num_pos[0])
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.s_encoder.hidden_size)

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

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_2(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
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
                        generate_input = torch.LongTensor([out_token]).to(self.device)

                        left_child, right_child, node_label = self.s_node_generater_2(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.s_merge_2(op.embedding, sub_stree.embedding, current_num)
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
        return beams[0].out,beams[0].score

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
        group_nums = batch_data['group nums']
        num_list = batch_data['num list']
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

        graphs = self.build_graph(seq_length, num_list, num_pos, group_nums)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0).to(self.device)
        batch_size = len(seq_length)
        seq_emb = self.t_embedder(seq)
        pade_outputs, _ = self.t_encoder(seq_emb, seq_length, graphs)
        problem_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        #print("encoder_outputs", encoder_outputs.size())
        #print("problem_output", problem_output.size())
        UNK_TOKEN = self.unk_token
        all_node_outputs=self.teacher_net_forward(encoder_outputs,problem_output,target,target_length,\
                                num_pos,nums_stack,padding_hidden,seq_mask,num_mask,UNK_TOKEN,num_start)
        #all_node_outputs = torch.stack(all_node_outputs, dim=1).to(self.device)
        all_node_outputs = torch.stack(all_node_outputs, dim=1)
        for id_,soft_target in zip(ques_id,all_node_outputs.split(1)):
            self.soft_target[id_]=soft_target.cpu()

    def init_encoder_mask(self,batch_size):
        encoder_mask = torch.FloatTensor(batch_size,115,self.hidden_size).uniform_() < 0.99
        self.encoder_mask = encoder_mask.float().to(self.device)

    def get_soft_target(self,batch_id):
        soft_tsrget=[]
        for id_ in batch_id:
            soft_tsrget.append(self.soft_target[id_])
        return soft_tsrget
    
    
    def soft_target_loss(self,logits, soft_target,length):
        if torch.cuda.is_available():
            length = torch.LongTensor(length).cuda()
        else:
            length = torch.LongTensor(length)
        loss_total = []
        for predict,label in zip(logits.split(1,dim=1),soft_target.split(1,dim=1)):
            predict = predict.squeeze()
            label = label.squeeze()
            loss_t = self.soft_cross_entropy_loss(predict,label)
            loss_total.append(loss_t)
        loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
        #loss_total = loss_total.sum(dim=1)
        loss_total = loss_total.sum() / length.float().sum()
        return loss_total
    
    def soft_cross_entropy_loss(self,predict_score,label_score):
        log_softmax = torch.nn.LogSoftmax(dim = 1)
        softmax = torch.nn.Softmax(dim = 1)

        predict_prob_log = log_softmax(predict_score).float()
        label_prob = softmax(label_score).float()


        loss_elem = -label_prob * predict_prob_log
        loss = loss_elem.sum(dim = 1)
        return loss
    
    def cosine_loss(self,logits, logits_1,length):
        if torch.cuda.is_available():
            length = torch.LongTensor(length).cuda()
        else:
            length = torch.LongTensor(length)
        loss_total = []
        for predict,label in zip(logits.split(1,dim=1),logits_1.split(1,dim=1)):
            predict = predict.squeeze()
            label = label.squeeze()
            loss_t = self.cosine_sim(predict,label)
            loss_total.append(loss_t)
        loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
        #loss_total = loss_total.sum(dim=1)
        loss_total = loss_total.sum() / length.float().sum()
        return loss_total
    
    def cosine_sim(self,logits, logits_1):
        device=logits.device
        return torch.ones(logits.size(0)).to(device) + torch.cosine_similarity(logits, logits_1, dim=1).to(device)

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