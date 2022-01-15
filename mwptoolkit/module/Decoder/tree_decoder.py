# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 11:11:31
# @File: tree_decoder.py


import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Attention.tree_attention import TreeAttention
from mwptoolkit.module.Attention.hierarchical_attention import HierarchicalAttention
from mwptoolkit.module.Layer.tree_layers import Score,Dec_LSTM
from mwptoolkit.module.Layer.tree_layers import ScoreModel,GateNN,TreeEmbeddingModel,DecomposeModel,NodeEmbeddingNode
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Strategy.beam_search import BeamNode
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens

class TreeDecoder(nn.Module):
    r'''
    Seq2tree decoder with Problem aware dynamic encoding
    '''
    def __init__(self, hidden_size, op_nums, generate_size, dropout=0.5):
        super(TreeDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings, encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs)  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight        
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, nums_mask)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class SARTreeDecoder(nn.Module):
    r'''
    Seq2tree decoder with Semantically-Aligned Regularization
    '''

    def __init__(self, hidden_size, op_nums, generate_size, dropout=0.5):
        super(SARTreeDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        
        self.saligned_attn = TreeAttention(hidden_size,hidden_size)
        self.encoder_linear1 = nn.Linear(hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)

        self.decoder_linear1 = nn.Linear(hidden_size, hidden_size)
        self.decoder_linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask):
        """
        Args:
            node_stacks (list): node stacks.
            left_childs (list): representation of left childs.
            encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size].
            num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
            padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
            seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
            mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size]

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                num_score, number score, shape [batch_size, number_size].
                op, operator score, shape [batch_size, operator_size].
                current_node, current node representation, shape [batch_size, 1, hidden_size].
                current_context, current context representation, shape [batch_size, 1, hidden_size].
                embedding_weight, embedding weight, shape [batch_size, number_size, hidden_size].
        """
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                node_emb = g * t
                current_node_temp.append(node_emb)

        current_node = torch.stack(current_node_temp)
        # sub_tree_emb = torch.stack(sub_tree_emb)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        # s_aligned_vector=self.attn(current_embeddings, encoder_outputs, seq_mask)

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight        
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, nums_mask)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        return num_score, op, current_node, current_context, embedding_weight

    def Semantically_Aligned_Regularization(self, subtree_emb, s_aligned_vector):
        """
        Args:
            subtree_emb (torch.Tensor):
            s_aligned_vector (torch.Tensor):

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                s_aligned_a
                s_aligned_d
        """
        #s_aligned_a = self.R2(torch.tanh(self.R1(s_aligned_vector)))
        #s_aligned_d = self.R2(torch.tanh(self.R1(subtree_emb)))
        s_aligned_a = self.encoder_linear2(torch.tanh(self.encoder_linear1(s_aligned_vector)))
        s_aligned_d = self.decoder_linear2(torch.tanh(self.decoder_linear1(subtree_emb)))
        return s_aligned_a, s_aligned_d


# class SARTreeDecoder(nn.Module):
#     r'''
#     Seq2tree decoder with Semantically-Aligned Regularization
#     '''
#     def __init__(self, hidden_size, op_nums, generate_size, dropout=0.5):
#         super(SARTreeDecoder, self).__init__()
# 
#         # Keep for reference
#         self.hidden_size = hidden_size
#         self.generate_size = generate_size
#         self.op_nums = op_nums
# 
#         # Define layers
#         self.dropout = nn.Dropout(dropout)
# 
#         self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, hidden_size))
# 
#         # for Computational symbols and Generated numbers
#         self.concat_l = nn.Linear(hidden_size, hidden_size)
#         self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
#         self.concat_lg = nn.Linear(hidden_size, hidden_size)
#         self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
# 
#         self.ops = nn.Linear(hidden_size * 2, op_nums)
# 
#         self.attn = TreeAttention(hidden_size, hidden_size)
#         self.score = Score(hidden_size * 2, hidden_size)
# 
#         self.R1=nn.Linear(hidden_size,hidden_size)
#         self.R2=nn.Linear(hidden_size,hidden_size)
# 
# 
#     def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask):
#         """
#         Args:
#             node_stacks (list): node stacks.
#             left_childs (list): representation of left childs.
#             encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size].
#             num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
#             padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
#             seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
#             mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size]
#         
#         Returns:
#             tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
#                 num_score, number score, shape [batch_size, number_size].
#                 op, operator score, shape [batch_size, operator_size].
#                 current_node, current node representation, shape [batch_size, 1, hidden_size].
#                 current_context, current context representation, shape [batch_size, 1, hidden_size].
#                 embedding_weight, embedding weight, shape [batch_size, number_size, hidden_size].
#         """
#         current_embeddings = []
# 
#         for st in node_stacks:
#             if len(st) == 0:
#                 current_embeddings.append(padding_hidden)
#             else:
#                 current_node = st[-1]
#                 current_embeddings.append(current_node.embedding)
# 
#         current_node_temp = []
#         for l, c in zip(left_childs, current_embeddings):
#             if l is None:
#                 c = self.dropout(c)
#                 g = torch.tanh(self.concat_l(c))
#                 t = torch.sigmoid(self.concat_lg(c))
#                 current_node_temp.append(g * t)
#             else:
#                 ld = self.dropout(l)
#                 c = self.dropout(c)
#                 g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
#                 t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
#                 node_emb=g * t
#                 current_node_temp.append(node_emb)
# 
#         current_node = torch.stack(current_node_temp)
#         #sub_tree_emb = torch.stack(sub_tree_emb)
# 
#         current_embeddings = self.dropout(current_node)
# 
#         current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
#         current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
#         #s_aligned_vector=self.attn(current_embeddings, encoder_outputs, seq_mask)
# 
#         # the information to get the current quantity
#         batch_size = current_embeddings.size(0)
#         # predict the output (this node corresponding to output(number or operator)) with PADE
# 
#         repeat_dims = [1] * self.embedding_weight.dim()
#         repeat_dims[0] = batch_size
#         embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
#         embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N
# 
#         leaf_input = torch.cat((current_node, current_context), 2)
#         leaf_input = leaf_input.squeeze(1)
#         leaf_input = self.dropout(leaf_input)
# 
#         # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
#         # max pooling the embedding_weight        
#         embedding_weight_ = self.dropout(embedding_weight)
#         num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, nums_mask)
# 
#         # num_score = nn.functional.softmax(num_score, 1)
# 
#         op = self.ops(leaf_input)
# 
#         return num_score, op, current_node, current_context, embedding_weight
# 
#     def Semantically_Aligned_Regularization(self,subtree_emb, s_aligned_vector):
#         """
#         Args:
#             subtree_emb (torch.Tensor):
#             s_aligned_vector (torch.Tensor):
# 
#         Returns:
#             tuple(torch.Tensor, torch.Tensor):
#                 s_aligned_a
#                 s_aligned_d
#         """
#         s_aligned_a=self.R2(torch.tanh(self.R1(s_aligned_vector)))
#         s_aligned_d=self.R2(torch.tanh(self.R1(subtree_emb)))
#         return s_aligned_a,s_aligned_d


class RNNBasedTreeDecoder(nn.Module):
    def __init__(self, input_size,embedding_size,hidden_size,dropout_ratio):
        super(RNNBasedTreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = BasicEmbedder(input_size, embedding_size,dropout_ratio, padding_idx=0)

        self.lstm = Dec_LSTM(embedding_size,hidden_size,dropout_ratio)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):

        src_emb = self.embedding(input_src)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h, parent_h, sibling_state)
        return prev_cy, prev_hy

class PredictModel(nn.Module):
    def __init__(self, hidden_size, class_size, dropout=0.4):
        super(PredictModel, self).__init__()
        self.class_size = class_size

        self.dropout = nn.Dropout(p=dropout)
        self.attn = HierarchicalAttention(hidden_size)
        
        self.score_pointer = ScoreModel(hidden_size)
        self.score_generator = ScoreModel(hidden_size)
        self.score_span = ScoreModel(hidden_size)  
        self.gen_prob = nn.Linear(hidden_size*2, 1)
        return
    
    def score_pn(self, hidden, context, embedding_masks):
        # embedding: batch_size * pointer_size * hidden_size
        # mask: batch_size * pointer_size
        device = hidden.device
        (pointer_embedding, pointer_mask), generator_embedding, _ = embedding_masks
        pointer_embedding = pointer_embedding.to(device)
        pointer_mask = pointer_mask.to(device)
        generator_embedding = generator_embedding.to(device)
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        pointer_embedding = self.dropout(pointer_embedding)
        pointer_score = self.score_pointer(hidden, context, pointer_embedding)
        pointer_score.data.masked_fill_(pointer_mask, -float('inf'))
        # batch_size * symbol_size
        # pointer
        pointer_prob = F.softmax(pointer_score, dim=-1)
        
        generator_embedding = self.dropout(generator_embedding)
        generator_score = self.score_generator(hidden, context, generator_embedding)
        # batch_size * generator_size
        # generator
        generator_prob = F.softmax(generator_score, dim=-1)
        # batch_size * class_size, softmax
        return pointer_prob, generator_prob

    def forward(self, node_hidden, encoder_outputs, masks, embedding_masks):
        use_cuda = node_hidden.is_cuda
        node_hidden_dropout = self.dropout(node_hidden).unsqueeze(1)
        span_output, word_outputs = encoder_outputs
        span_mask, word_masks = masks
        if use_cuda:
            span_mask = span_mask.cuda()
            word_masks = [mask.cuda() for mask in word_masks]
        output_attn = self.attn(node_hidden_dropout, span_output, word_outputs, span_mask, word_masks)
        context = output_attn.squeeze(1)

        hc = torch.cat((node_hidden, context), dim=-1)
        # log(f(softmax(x)))
        # prob: softmax
        pointer_prob, generator_prob = self.score_pn(node_hidden, context, embedding_masks)
        gen_prob = torch.sigmoid(self.gen_prob(hc))
        prob = torch.cat((gen_prob * generator_prob, (1 - gen_prob) * pointer_prob), dim=-1)
        # batch_size * class_size
        # generator + pointer + empty_pointer
        pad_empty_pointer = torch.zeros(prob.size(0), self.class_size - prob.size(-1))
        if use_cuda:
            pad_empty_pointer = pad_empty_pointer.cuda()
        prob = torch.cat((prob, pad_empty_pointer), dim=-1)
        output = torch.log(prob + 1e-30)
        return output, context

class HMSDecoder(nn.Module):
    def __init__(self, embedding_model, hidden_size, dropout, op_set, vocab_dict, class_list, device):
        super(HMSDecoder, self).__init__()
        self.hidden_size = hidden_size
        #self.use_cuda = use_cuda
        embed_size = embedding_model.embedding_size
        class_size = len(class_list)

        self.get_predict_meta(class_list, vocab_dict, device)

        self.embed_model = embedding_model
        # 128 => 512
        self.op_hidden = nn.Linear(embed_size, hidden_size)
        self.predict = PredictModel(hidden_size, class_size, dropout=dropout)
        op_set = set(i for i, symbol in enumerate(class_list) if symbol in op_set)
        self.tree_embedding = TreeEmbeddingModel(hidden_size, op_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_size, dropout, device)
        return

    def get_predict_meta(self, class_list, vocab_dict, device):
        # embed order: generator + pointer, with original order
        # used in predict_model, tree_embedding
        pointer_list = [token for token in class_list if (token in NumMask.number) or (token == SpecialTokens.UNK_TOKEN)]
        generator_list = [token for token in class_list if token not in pointer_list]
        embed_list = generator_list + pointer_list

        # pointer num index in class_list, for select only num pos from num_pos with op pos
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        # generator symbol index in vocab, for generator symbol embedding
        self.generator_vocab = torch.LongTensor([vocab_dict[token] for token in generator_list])
        # class_index -> embed_index, for tree embedding
        # embed order -> class order, for predict_model output
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
    
        self.pointer_index = self.pointer_index.to(device)
        self.generator_vocab = self.generator_vocab.to(device)
        self.class_to_embed_index = self.class_to_embed_index.to(device)
        return

    def get_pad_masks(self, encoder_outputs, input_lengths, span_length=None):
        span_output, word_outputs = encoder_outputs
        span_pad_length = span_output.size(1)
        word_pad_lengths = [word_output.size(1) for word_output in word_outputs]
        
        span_mask = self.get_mask(span_length, span_pad_length)
        word_masks = [self.get_mask(input_length, word_pad_length) for input_length, word_pad_length in zip(input_lengths, word_pad_lengths)]
        masks = (span_mask, word_masks)
        return masks
    def get_mask(self, encode_lengths, pad_length):
        device = encode_lengths.device
        batch_size = encode_lengths.size(0)
        index = torch.arange(pad_length).to(device)

        mask = (index.unsqueeze(0).expand(batch_size, -1) >= encode_lengths.unsqueeze(-1)).byte()
        # save one position for full padding span to prevent nan in softmax
        # invalid value in full padding span will be ignored in span level attention
        mask[mask.sum(dim=-1) == pad_length, 0] = 0
        return mask
    
    def get_pointer_meta(self, num_pos, sub_num_poses=None):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        # length of word_num_poses determined by span_num_pos
        if sub_num_poses is not None:
            sub_pointer_poses = [sub_num_pos.index_select(dim=1, index=self.pointer_index)[:, :occupied_len] for sub_num_pos in sub_num_poses]
        else:
            sub_pointer_poses = None
        return pointer_num_pos, sub_pointer_poses

    def get_pointer_embedding(self, pointer_num_pos, encoder_outputs):
        # encoder_outputs: batch_size * seq_len * hidden_size
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        device = encoder_outputs.device
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size)
        batch_index = batch_index.to(device)
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        # batch_size * pointer_len * hidden_size
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        # mask invalid pos -1
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        return pointer_embedding
    
    def get_pointer_mask(self, pointer_num_pos):
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        pointer_mask = pointer_num_pos == -1
        return pointer_mask
    
    def get_generator_embedding_mask(self, batch_size):
        # generator_size * hidden_size
        generator_embedding = self.op_hidden(self.embed_model(self.generator_vocab))
        # batch_size * generator_size * hidden_size
        generator_embedding = generator_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_size * generator_size
        generator_mask = (self.generator_vocab == -1).unsqueeze(0).expand(batch_size, -1)
        return generator_embedding, generator_mask
    
    def get_class_embedding_mask(self, num_pos, encoder_outputs):
        # embedding: batch_size * size * hidden_size
        # mask: batch_size * size
        _, word_outputs = encoder_outputs
        span_num_pos, word_num_poses = num_pos
        generator_embedding, generator_mask = self.get_generator_embedding_mask(span_num_pos.size(0))
        span_pointer_num_pos, word_pointer_num_poses = self.get_pointer_meta(span_num_pos, word_num_poses)
        pointer_mask = self.get_pointer_mask(span_pointer_num_pos)
        num_pointer_embeddings = []
        for word_output, word_pointer_num_pos in zip(word_outputs, word_pointer_num_poses):
            num_pointer_embedding = self.get_pointer_embedding(word_pointer_num_pos, word_output)
            num_pointer_embeddings.append(num_pointer_embedding)
        pointer_embedding = torch.cat([embedding.unsqueeze(0) for embedding in num_pointer_embeddings], dim=0).sum(dim=0)
        
        all_embedding = torch.cat((generator_embedding, pointer_embedding), dim=1)
        pointer_embedding_mask = (pointer_embedding, pointer_mask)
        return pointer_embedding_mask, generator_embedding, all_embedding

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        return node_stacks, tree_stacks

    def forward_step(self, node_stacks, tree_stacks, nodes_hidden, encoder_outputs, masks, embedding_masks, decoder_nodes_class=None):
        nodes_output, nodes_context = self.predict(nodes_hidden, encoder_outputs, masks, embedding_masks)
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        predict_nodes_class = nodes_output.topk(1)[1]
        # teacher
        if decoder_nodes_class is not None:
            nodes_class = decoder_nodes_class.view(-1)
        # no teacher
        else:
            nodes_class = predict_nodes_class.view(-1)
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(embedding_masks[-1], tree_stacks, embed_nodes_index)
        nodes_hidden = self.decompose(node_stacks, tree_stacks, nodes_context, labels_embedding)
        return nodes_output, predict_nodes_class, nodes_hidden
    
    def forward_teacher(self, decoder_nodes_label, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden = self.forward_step(node_stacks, tree_stacks, decoder_hidden, encoder_outputs, masks, embedding_masks, decoder_nodes_class=decoder_node_class)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_beam(self, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width=1):
        # only support batch_size == 1
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        beams = [BeamNode(0, decoder_init_hidden, node_stacks, tree_stacks, [], [])]
        for _ in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                # finished stack-guided decoding
                if len(b.node_stacks) == 0:
                    current_beams.append(b)
                    continue
                # unfinished decoding
                # batch_size == 1
                # batch_size * class_size
                nodes_output, nodes_context = self.predict(b.nodes_hidden, encoder_outputs, masks, embedding_masks)
                nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                # batch_size * beam_width
                top_value, top_index = nodes_output.topk(beam_width)
                top_value = torch.exp(top_value)
                for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                    nb = b.copy()
                    embed_nodes_index = self.class_to_embed_index[predicted_symbol.view(-1)]
                    labels_embedding = self.tree_embedding(embedding_masks[-1], nb.tree_stacks, embed_nodes_index)
                    nodes_hidden = self.decompose(nb.node_stacks, nb.tree_stacks, nodes_context, labels_embedding, pad_node=False)

                    nb.score = b.score + predict_score.item()
                    nb.nodes_hidden = nodes_hidden
                    nb.decoder_outputs_list.append(nodes_output)
                    nb.sequence_symbols_list.append(predicted_symbol)
                    current_beams.append(nb)
            beams = sorted(current_beams, key=lambda b:b.score, reverse=True)
            beams = beams[:beam_width]
            all_finished = True
            for b in beams:
                if len(b.node_stacks[0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                break
        output = beams[0]
        return output.decoder_outputs_list, output.nodes_hidden, output.sequence_symbols_list

    def forward(self, targets=None, encoder_hidden=None, encoder_outputs=None, input_lengths=None, span_length=None, num_pos=None, max_length=None, beam_width=None):
        masks = self.get_pad_masks(encoder_outputs, input_lengths, span_length)
        embedding_masks = self.get_class_embedding_mask(num_pos, encoder_outputs)

        if type(encoder_hidden) is tuple:
            encoder_hidden = encoder_hidden[0]
        decoder_init_hidden = encoder_hidden[-1,:,:]

        if max_length is None:
            if targets is not None:
                max_length = targets.size(1)
            else:
                max_length = 40
        
        if beam_width is not None:
            return self.forward_beam(decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width)
        else:
            return self.forward_teacher(targets, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length)


class LSTMBasedTreeDecoder(nn.Module):
    r'''
    '''
    def __init__(self, embedding_size, hidden_size, op_nums, generate_size, dropout=0.5):
        super(LSTMBasedTreeDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, embedding_size))

        
        self.rnn = nn.LSTMCell(embedding_size*2+hidden_size, hidden_size) #
        self.tree_rnn = nn.LSTMCell(embedding_size*2+hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size, op_nums)
        self.trans = nn.Linear(hidden_size, embedding_size)

        self.attention = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size, embedding_size)

        self.p_z = nn.Linear(hidden_size, 1)
        self.copy_attention = TreeAttention(hidden_size, hidden_size)

    def forward(self, parent_embed, left_embed, prev_embed, encoder_outputs, num_pades, padding_hidden,
                seq_mask, nums_mask, hidden, tree_hidden):
        """
        Args:
            parent_embed (list): parent embedding, length [batch_size], list of torch.Tensor with shape [1, 2 * hidden_size].
            left_embed (list): left embedding, length [batch_size], list of torch.Tensor with shape [1, embedding_size].
            prev_embed (list): previous embedding, length [batch_size], list of torch.Tensor with shape [1, embedding_size].
            encoder_outputs (torch.Tensor): output from encoder, shape [batch_size, sequence_length, hidden_size].
            num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
            padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
            seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
            mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size].
            hidden (tuple(torch.Tensor, torch.Tensor)): hidden states, shape [batch_size, num_directions * hidden_size].
            tree_hidden (tuple(torch.Tensor, torch.Tensor)): tree hidden states, shape [batch_size, num_directions * hidden_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                num_score, number score, shape [batch_size, number_size].
                op, operator score, shape [batch_size, operator_size].
                current_embeddings, current node representation, shape [batch_size, 1, num_directions * hidden_size].
                current_context, current context representation, shape [batch_size, 1, num_directions * hidden_size].
                embedding_weight, embedding weight, shape [batch_size, number_size, embedding_size].
                hidden (tuple(torch.Tensor, torch.Tensor)): hidden states, shape [batch_size, num_directions * hidden_size].
                tree_hidden (tuple(torch.Tensor, torch.Tensor)): tree hidden states, shape [batch_size, num_directions * hidden_size].
        """
        parent_embed = torch.cat(parent_embed, dim=0)
        left_embed = torch.cat(left_embed, dim=0)
        prev_embed = torch.cat(prev_embed, dim=0)
        batch_size = parent_embed.size(0)

        embedded = torch.cat([parent_embed, left_embed, prev_embed], dim=1)
        #print('embedded', embedded.size(), len(hidden), hidden[0].size())
        if hidden[0].size(0) != batch_size:
            hidden = (hidden[0].repeat(batch_size, 1), hidden[1].repeat(batch_size, 1))
        #print(hidden[0].size(), batch_size, embedded.size())
        hidden_h, hidden_c = self.rnn(embedded, hidden) #self.rnn(embedded, hidden)
        hidden = (hidden_h, hidden_c)

        if tree_hidden[0].size(0) != batch_size:
            tree_hidden = (tree_hidden[0].repeat(batch_size, 1), tree_hidden[1].repeat(batch_size, 1))
        tree_hidden_h, tree_hidden_c = self.tree_rnn(embedded, tree_hidden)
        tree_hidden = (tree_hidden_h, tree_hidden_c)
        output = self.linear(torch.cat((hidden_h, tree_hidden_h), dim=-1)).unsqueeze(1)
        
        if encoder_outputs.size(0) != batch_size:
            repeat_dims = [1] * encoder_outputs.dim()
            repeat_dims[0] = batch_size
            encoder_outputs = encoder_outputs.repeat(*repeat_dims)
        
        current_attn = self.attention(output.transpose(0,1), encoder_outputs.transpose(0,1), seq_mask)
        output = current_attn.bmm(encoder_outputs)

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        
        embedding_weight = torch.cat((embedding_weight, self.trans(num_pades)), dim=1)  # B x O x N

        embedding_weight_ = self.dropout(embedding_weight)
        
        num_score = self.score(output, embedding_weight_, nums_mask)
        op = self.ops(output.squeeze(1))

        return num_score, op, output, output, embedding_weight, hidden, tree_hidden
