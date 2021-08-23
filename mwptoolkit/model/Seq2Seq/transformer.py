# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:38:29
# @File: transformer.py

import random

import torch
from torch import nn

from mwptoolkit.module.Encoder.transformer_encoder import TransformerEncoder
from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder
from mwptoolkit.module.Embedder.position_embedder import PositionEmbedder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Attention.self_attention import SelfAttentionMask
from mwptoolkit.module.Strategy.beam_search import Beam_Search_Hypothesis
from mwptoolkit.module.Strategy.sampling import topk_sampling
from mwptoolkit.module.Strategy.greedy import greedy_search
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens
from mwptoolkit.module.Decoder.rnn_decoder import BasicRNNDecoder, AttentionalRNNDecoder


class Transformer(nn.Module):
    """
    Reference:
        Vaswani et al. "Attention Is All You Need".
    """
    def __init__(self, config, dataset):
        super().__init__()
        self.max_output_len = config["max_output_len"]
        self.share_vocab = config["share_vocab"]
        self.decoding_strategy = config["decoding_strategy"]
        self.teacher_force_ratio = config["teacher_force_ratio"]

        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]

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
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)

        self.in_embedder = BaiscEmbedder(self.vocab_size, config["embedding_size"], config["embedding_dropout_ratio"])
        if config["share_vocab"]:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BaiscEmbedder(self.symbol_size, config["embedding_size"], config["embedding_dropout_ratio"])

        #self.pos_embedder=PositionEmbedder(config["embedding_size"],config["device"],config["embedding_dropout_ratio"],config["max_len"])
        self.pos_embedder = PositionEmbedder(config["embedding_size"], config["max_len"])
        self.self_attentioner = SelfAttentionMask()
        self.encoder=TransformerEncoder(config["embedding_size"],config["ffn_size"],config["num_encoder_layers"],\
                                            config["num_heads"],config["attn_dropout_ratio"],\
                                            config["attn_weight_dropout_ratio"],config["ffn_dropout_ratio"])
        self.decoder=TransformerDecoder(config["embedding_size"],config["ffn_size"],config["num_decoder_layers"],\
                                            config["num_heads"],config["attn_dropout_ratio"],\
                                            config["attn_weight_dropout_ratio"],config["ffn_dropout_ratio"])
        # self.decoder = BasicRNNDecoder(config["embedding_size"], 128, 1, \
        #                                'lstm', config["attn_dropout_ratio"])
        self.generate_linear = nn.Linear(128, self.symbol_size)
        self.out = nn.Linear(config["embedding_size"], self.symbol_size)

        weight = torch.ones(self.symbol_size).to(config["device"])
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, src, target=None):
        print('src', src)
        exit()
        #source_embeddings = self.pos_embedder(self.in_embedder(src))
        device = src.device
        source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
        source_padding_mask = torch.eq(src, self.out_pad_token)
        encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)

        if target != None:
            token_logits = self.generate_t(target, encoder_outputs, source_padding_mask)
            return token_logits
        else:
            all_outputs = self.generate_without_t(encoder_outputs, source_padding_mask)
            return all_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).to(device).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def calculate_loss(self, batch_data):
        """Finish forward-propagating, calculating loss and back-propagation.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            float: loss value.
        """
        src = batch_data['question']
        target = batch_data['equation']
        device = src.device
        source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
        source_padding_mask = torch.eq(src, self.out_pad_token)
        encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)
        #print('encoder_outputs', encoder_outputs.size())

        # decoder_inputs = self.init_decoder_inputs(target, device, encoder_outputs.size(0))
        # #print('decoder_inputs', decoder_inputs.size())
        # token_logits = self.generate_t(target, (encoder_outputs[:, -1, :].unsqueeze(0).contiguous(), encoder_outputs[:, -1, :].unsqueeze(0).contiguous()), decoder_inputs)
        token_logits = self.generate_t(target, encoder_outputs, source_padding_mask)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        self.loss.reset()
        self.loss.eval_batch(token_logits, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data):
        """Model test.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        src = batch_data['question']
        target = batch_data['equation']
        num_list = batch_data['num list']

        device = src.device
        source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
        source_padding_mask = torch.eq(src, self.out_pad_token)
        encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)

        # decoder_inputs = self.init_decoder_inputs(target=None, device=device, batch_size=encoder_outputs.size(0))
        # all_outputs = self.generate_without_t(target, (encoder_outputs[:, -1, :].unsqueeze(0).contiguous(), encoder_outputs[:, -1, :].unsqueeze(0).contiguous()), decoder_inputs)
        all_outputs = self.generate_without_t(encoder_outputs, source_padding_mask)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(all_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def generate_t_tmp(self, encoder_outputs, encoder_hidden, decoder_inputs):
        with_t = random.random()
        if with_t < self.teacher_force_ratio:
            #print('decoder_inputs', encoder_hidden[0].size(), encoder_hidden[1].size())
            decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
            token_logits = torch.nn.functional.log_softmax(token_logits, dim=1)
            #token_logits=torch.log_softmax(token_logits,dim=1)
        else:
            seq_len = decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            token_logits = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                predict = torch.nn.functional.log_softmax(token_logit, dim=1)
                #predict=torch.log_softmax(token_logit,dim=1)
                output = predict.topk(1, dim=1)[1]
                token_logits.append(predict)

                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
            token_logits = torch.stack(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def generate_t(self, target, encoder_outputs, source_padding_mask):
        with_t = random.random()
        seq_len = target.size(1)
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, -1).to(device)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]

            #decoder_inputs = self.pos_embedder(self.out_embedder(target))
            decoder_inputs = self.out_embedder(target) + self.pos_embedder(target).to(device)
            self_padding_mask = torch.eq(target, self.out_pad_token)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            # print('decoder_inputs', decoder_inputs.size(), self_padding_mask.size())
            # print(self_padding_mask[:3, :])
            # print('self_attn_mask', self_attn_mask.size(), encoder_outputs.size(), source_padding_mask.size())
            # print(self_attn_mask[:5, :5], source_padding_mask[:3, :50]);
            decoder_outputs = self.decoder(decoder_inputs,
                                           self_padding_mask=self_padding_mask,
                                           self_attn_mask=self_attn_mask,
                                           external_states=encoder_outputs,
                                           external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        else:
            token_logits = []
            input_seq = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, -1).to(device)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                #decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq).to(device)
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)

                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                #output=greedy_search(token_logit)
                output = torch.topk(token_logit.squeeze(), 1, dim=-1)[1]
                if self.share_vocab:
                    pre_tokens.append(self.convert_out_idx_2_in_idx(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        token_logits = torch.log_softmax(token_logits, dim=1)
        return token_logits

    def generate_without_t_tmp(self, encoder_outputs, encoder_hidden, decoder_input):
        all_outputs = []
        decoder_hidden = encoder_hidden
        for idx in range(30):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            token_logits = self.generate_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logits, dim=1)
            output = predict.topk(1, dim=1)[1]

            all_outputs.append(output)
            if self.share_vocab:
                output = self.convert_out_idx_2_in_idx(output)
                decoder_input = self.out_embedder(output)
            else:
                decoder_input = self.out_embedder(output)
        all_outputs = torch.cat(all_outputs, dim=1)
        return all_outputs

    def generate_without_t(self, encoder_outputs, source_padding_mask):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        input_seq = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, -1).to(device)
        pre_tokens = [input_seq]
        all_outputs = []
        for gen_idx in range(self.max_output_len):
            self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
            decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq).to(device)
            #decoder_input = self.pos_embedder(self.out_embedder(input_seq))
            decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)

            token_logits = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
            if self.decoding_strategy == "topk_sampling":
                output = topk_sampling(token_logits, top_k=5)
            elif self.decoding_strategy == "greedy_search":
                output = greedy_search(token_logits)
            else:
                raise NotImplementedError
            all_outputs.append(output)
            if self.share_vocab:
                pre_tokens.append(self.convert_out_idx_2_in_idx(output))
            else:
                pre_tokens.append(output)
            input_seq = torch.cat(pre_tokens, dim=1)
        all_outputs = torch.cat(all_outputs, dim=1)
        return all_outputs

    def decode(self, output):
        device = output.device

        batch_size, seq_len = output.size()
        decoded_output = []
        for b_i in range(batch_size):
            b_output = []
            for idx in range(seq_len):
                b_output.append(self.in_word2idx[self.out_idx2symbol[output[b_i, idx]]])
            decoded_output.append(b_output)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_out_idx_2_in_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self) -> str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters


# class Transformer(nn.Module):
#     def __init__(self, config, dataset):
#         super().__init__()
#         self.max_output_len = config["max_output_len"]
#         self.share_vocab = config["share_vocab"]
#         self.decoding_strategy = config["decoding_strategy"]
#         self.teacher_force_ratio = config["teacher_force_ratio"]

#         self.mask_list = NumMask.number
#         self.out_symbol2idx = config["out_symbol2idx"]
#         self.out_idx2symbol = config["out_idx2symbol"]
#         self.in_word2idx = config["in_word2idx"]
#         self.in_idx2word = config["in_idx2word"]
#         self.in_pad_idx = self.in_word2idx[SpecialTokens.PAD_TOKEN]

#         if config["share_vocab"]:
#             self.out_pad_idx = self.in_pad_idx
#             self.out_sos_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
#         else:
#             self.out_pad_idx = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
#             self.out_sos_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]

#         try:
#             self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
#         except:
#             self.out_sos_token = None
#         try:
#             self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
#         except:
#             self.out_eos_token = None
#         try:
#             self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
#         except:
#             self.out_pad_token = None

#         self.in_embedder = BaiscEmbedder(config["vocab_size"], config["embedding_size"], config["embedding_dropout_ratio"])
#         if config["share_vocab"]:
#             self.out_embedder = self.in_embedder
#         else:
#             self.out_embedder = BaiscEmbedder(config["symbol_size"], config["embedding_size"], config["embedding_dropout_ratio"])

#         #self.pos_embedder=PositionEmbedder(config["embedding_size"],config["device"],config["embedding_dropout_ratio"],config["max_len"])
#         self.pos_embedder = PositionEmbedder(config["embedding_size"], config["max_len"])
#         self.self_attentioner = SelfAttentionMask()
#         self.encoder=TransformerEncoder(config["embedding_size"],config["ffn_size"],config["num_encoder_layers"],\
#                                             config["num_heads"],config["attn_dropout_ratio"],\
#                                             config["attn_weight_dropout_ratio"],config["ffn_dropout_ratio"])
#         self.decoder=TransformerDecoder(config["embedding_size"],config["ffn_size"],config["num_decoder_layers"],\
#                                             config["num_heads"],config["attn_dropout_ratio"],\
#                                             config["attn_weight_dropout_ratio"],config["ffn_dropout_ratio"])
#         self.out = nn.Linear(config["embedding_size"], config["symbol_size"])

#         weight = torch.ones(config['symbol_size']).to(config["device"])
#         pad = self.out_pad_token
#         self.loss = NLLLoss(weight, pad)

#     def forward(self, src, target=None):
#         #source_embeddings = self.pos_embedder(self.in_embedder(src))
#         device = src.device
#         source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
#         source_padding_mask = torch.eq(src, self.in_pad_idx)
#         encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)

#         if target != None:
#             token_logits = self.generate_t(target, encoder_outputs, source_padding_mask)
#             return token_logits
#         else:
#             all_outputs = self.generate_without_t(encoder_outputs, source_padding_mask)
#             return all_outputs

#     def calculate_loss(self, batch_data):
#         src = batch_data['question']
#         target = batch_data['equation']
#         device = src.device
#         source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
#         source_padding_mask = torch.eq(src, self.in_pad_idx)
#         encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)

#         token_logits = self.generate_t(target, encoder_outputs, source_padding_mask)
#         if self.share_vocab:
#             target = self.convert_in_idx_2_out_idx(target)
#         self.loss.reset()
#         self.loss.eval_batch(token_logits, target.view(-1))
#         self.loss.backward()
#         return self.loss.get_loss()

#     def model_test(self, batch_data):
#         src = batch_data['question']
#         target = batch_data['equation']
#         num_list = batch_data['num list']

#         device = src.device
#         source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
#         source_padding_mask = torch.eq(src, self.in_pad_idx)
#         encoder_outputs = self.encoder(source_embeddings, self_padding_mask=source_padding_mask)

#         all_outputs = self.generate_without_t(encoder_outputs, source_padding_mask)
#         if self.share_vocab:
#             target = self.convert_in_idx_2_out_idx(target)
#         all_outputs = self.convert_idx2symbol(all_outputs, num_list)
#         targets = self.convert_idx2symbol(target, num_list)
#         return all_outputs, targets

#     def generate_t(self, target, encoder_outputs, source_padding_mask):
#         with_t = random.random()
#         seq_len = target.size(1)
#         batch_size = encoder_outputs.size(0)
#         device = encoder_outputs.device
#         if with_t < self.teacher_force_ratio:
#             input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
#             target = torch.cat((input_seq, target), dim=1)[:, :-1]

#             #decoder_inputs = self.pos_embedder(self.out_embedder(target))
#             decoder_inputs = self.out_embedder(target) + self.pos_embedder(target).to(device)
#             self_padding_mask = torch.eq(target, self.out_pad_idx)
#             self_attn_mask = self.self_attentioner(target.size(-1)).bool()
#             decoder_outputs = self.decoder(decoder_inputs,
#                                            self_padding_mask=self_padding_mask,
#                                            self_attn_mask=self_attn_mask,
#                                            external_states=encoder_outputs,
#                                            external_padding_mask=source_padding_mask)
#             token_logits = self.out(decoder_outputs)
#             token_logits = token_logits.view(-1, token_logits.size(-1))
#         else:
#             token_logits = []
#             input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
#             pre_tokens = [input_seq]
#             for idx in range(seq_len):
#                 self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
#                 #decoder_input = self.pos_embedder(self.out_embedder(input_seq))
#                 decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq).to(device)
#                 decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)

#                 token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
#                 token_logits.append(token_logit)
#                 #output=greedy_search(token_logit)
#                 output = torch.topk(token_logit.squeeze(), 1, dim=-1)[1]
#                 if self.share_vocab:
#                     pre_tokens.append(self.convert_out_idx_2_in_idx(output))
#                 else:
#                     pre_tokens.append(output)
#                 input_seq = torch.cat(pre_tokens, dim=1)
#             token_logits = torch.cat(token_logits, dim=1)
#             token_logits = token_logits.view(-1, token_logits.size(-1))
#         token_logits = torch.log_softmax(token_logits, dim=1)
#         return token_logits

#     def generate_without_t(self, encoder_outputs, source_padding_mask):
#         batch_size = encoder_outputs.size(0)
#         device = encoder_outputs.device
#         input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
#         pre_tokens = [input_seq]
#         all_outputs = []
#         for gen_idx in range(self.max_output_len):
#             self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
#             decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq).to(device)
#             #decoder_input = self.pos_embedder(self.out_embedder(input_seq))
#             decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)

#             token_logits = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
#             if self.decoding_strategy == "topk_sampling":
#                 output = topk_sampling(token_logits, top_k=5)
#             elif self.decoding_strategy == "greedy_search":
#                 output = greedy_search(token_logits)
#             else:
#                 raise NotImplementedError
#             all_outputs.append(output)
#             if self.share_vocab:
#                 pre_tokens.append(self.convert_out_idx_2_in_idx(output))
#             else:
#                 pre_tokens.append(output)
#             input_seq = torch.cat(pre_tokens, dim=1)
#         all_outputs = torch.cat(all_outputs, dim=1)
#         return all_outputs

#     def decode(self, output):
#         device = output.device

#         batch_size, seq_len = output.size()
#         decoded_output = []
#         for b_i in range(batch_size):
#             b_output = []
#             for idx in range(seq_len):
#                 b_output.append(self.in_word2idx[self.out_idx2symbol[output[b_i, idx]]])
#             decoded_output.append(b_output)
#         decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
#         return decoded_output

#     def convert_out_idx_2_in_idx(self, output):
#         device = output.device

#         batch_size = output.size(0)
#         seq_len = output.size(1)

#         decoded_output = []
#         for b_i in range(batch_size):
#             output_i = []
#             for s_i in range(seq_len):
#                 output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
#             decoded_output.append(output_i)
#         decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
#         return decoded_output

#     def convert_in_idx_2_out_idx(self, output):
#         device = output.device

#         batch_size = output.size(0)
#         seq_len = output.size(1)

#         decoded_output = []
#         for b_i in range(batch_size):
#             output_i = []
#             for s_i in range(seq_len):
#                 output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
#             decoded_output.append(output_i)
#         decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
#         return decoded_output

#     def convert_idx2symbol(self, output, num_list):
#         batch_size = output.size(0)
#         seq_len = output.size(1)
#         output_list = []
#         for b_i in range(batch_size):
#             res = []
#             num_len = len(num_list[b_i])
#             for s_i in range(seq_len):
#                 idx = output[b_i][s_i]
#                 if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
#                     break
#                 symbol = self.out_idx2symbol[idx]
#                 if "NUM" in symbol:
#                     num_idx = self.mask_list.index(symbol)
#                     if num_idx >= num_len:
#                         res.append(symbol)
#                     else:
#                         res.append(num_list[b_i][num_idx])
#                 else:
#                     res.append(symbol)
#             output_list.append(res)
#         return output_list

#     def __str__(self) -> str:
#         info = super().__str__()
#         total = sum(p.numel() for p in self.parameters())
#         trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
#         return info + parameters
