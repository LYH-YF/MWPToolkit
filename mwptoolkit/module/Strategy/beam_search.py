# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:12:20
# @File: beam_search.py


import copy
import torch
from torch.nn import functional as F
from mwptoolkit.utils.utils import copy_list


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden,token_logits,outputs, all_output=None):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output
        self.token_logits = token_logits
        self.outputs = outputs


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out, token_logit=None):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.token_logit = token_logit


class BeamNode:
    def __init__(self, score, nodes_hidden, node_stacks, tree_stacks, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return
    
    def copy(self):
        node = BeamNode(
            self.score,
            self.nodes_hidden,
            copy_list(self.node_stacks),
            copy_list(self.tree_stacks),
            copy_list(self.decoder_outputs_list),
            copy_list(self.sequence_symbols_list)
        )
        return node

class Beam_Search_Hypothesis(object):
    r""" Class designed for beam search.
    """
    def __init__(self, beam_size, sos_token_idx, eos_token_idx, device, idx2token):
        self.beam_size = beam_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.device = device
        self.idx2token = idx2token

        self.hypthetic_token_idx = [[sos_token_idx]]
        self.completed_hypotheses = []
        self.hyp_scores = torch.zeros(1).to(device)
    
    def generate(self):
        r""" Pick the hypothesis with max prob among beam_size hypothesises.

        Return:
            List[str]: the generated tokens
        """
        generate_idx = self.hypthetic_token_idx[0][1:] if (len(self.completed_hypotheses) == 0) else max(self.completed_hypotheses, key = lambda hyp: hyp[1])[0]
        generate_tokens = [self.idx2token[idx.item()] for idx in generate_idx]
        return generate_tokens
    
    def stop(self):
        r""" Determine if the beam search is over.

        Return:
            Bool: ``True`` represents the search over, `Flase` represents the search working.
        """
        return len(self.completed_hypotheses) == self.beam_size
    
    def step(self, gen_idx, token_logits, decoder_states=None, encoder_output=None, encoder_mask=None, input_type='token'):
        r""" A step for beam search.

        Args:
            gen_idx (int): the generated step number.
            token_logits (torch.Tensor): logits distribution, shape: [hyp_num, sequence_length, vocab_size].
            decoder_states (torch.Tensor, optional): the states of decoder needed to choose, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_output (torch.Tensor, optional): the output of encoder needed to copy, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_mask (torch.Tensor, optional): the mask of encoder to copy, shape: [hyp_num, sequence_length], default: None.

        Return:
            torch.Tensor: the next input squence, shape: [hyp_num],
            torch.Tensor, optional: the chosen states of decoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed output of encoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed mask of encoder, shape: [new_hyp_num, sequence_length]
        """
        token_probs = F.log_softmax(token_logits, dim=-1).squeeze(1)
        vocab_size = token_probs.shape[-1]

        live_hyp_num = self.beam_size - len(self.completed_hypotheses)
        tmp_hyp_scores = (self.hyp_scores.unsqueeze(1).expand_as(token_probs) + token_probs).view(-1)
        top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)
        hyp_ids = top_pos // vocab_size
        word_ids = top_pos % vocab_size

        new_hypotheses = []
        new_ids = []
        new_scores = []

        for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
            new_hyp = self.hypthetic_token_idx[hyp_id] + [word_id]
            if (word_id == self.eos_token_idx):
                self.completed_hypotheses.append((new_hyp[1:-1], score / (gen_idx - 1)))
            else:
                new_hypotheses.append(new_hyp)
                new_ids.append(hyp_id)
                new_scores.append(score)

        if (len(self.completed_hypotheses) == self.beam_size):
            none_cnt = (decoder_states is not None) + (encoder_output is not None) + (encoder_mask is not None) + 1
            return [None] * none_cnt

        self.hypthetic_token_idx = new_hypotheses
        self.hyp_scores = torch.tensor(new_scores).to(self.device)

        hyp_num = len(self.hypthetic_token_idx)
        if (input_type == 'token'):
            input_seq = [hyp[-1] for hyp in self.hypthetic_token_idx]
            input_seq = torch.tensor(input_seq).unsqueeze(1).to(self.device)
        elif (input_type == 'whole'):
            input_seq = torch.tensor(self.hypthetic_token_idx).to(self.device)
        else:
            raise ValueError("The input type must be in ['token', 'whole'].")

        returns = [input_seq]

        if (decoder_states is not None):
            new_ids = torch.tensor(new_ids).to(self.device)
            decoder_states = decoder_states[:, new_ids, :]
            returns += [decoder_states]

        if (encoder_output is not None):
            encoder_output = encoder_output[0:1].repeat(hyp_num, 1, 1)
            encoder_mask = encoder_mask[0:1].repeat(hyp_num, 1)
            returns += [encoder_output, encoder_mask]
            
        return returns
