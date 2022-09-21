# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:38:29
# @File: transformer.py

import random
from typing import Tuple, Dict, Any

import torch
from torch import nn

from mwptoolkit.module.Encoder.transformer_encoder import TransformerEncoder
from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder
from mwptoolkit.module.Embedder.position_embedder import PositionEmbedder
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
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
        super(Transformer, self).__init__()
        self.device = config['device']
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

        self.in_embedder = BasicEmbedder(self.vocab_size, config["embedding_size"], config["embedding_dropout_ratio"])
        if config["share_vocab"]:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, config["embedding_size"], config["embedding_dropout_ratio"])

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

    def forward(self, src, target=None,output_all_layers=False) -> Tuple[
            torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor src: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor|None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: default False, return output of all layers if output_all_layers is True.
        :return: token_logits, symbol_outputs, model_all_outputs.
        :rtype tuple(torch.Tensor, torch.Tensor, dict)
        """
        device = src.device
        source_embeddings = self.in_embedder(src) + self.pos_embedder(src).to(device)
        source_padding_mask = torch.eq(src, self.out_pad_token)

        encoder_outputs,encoder_layer_outputs = self.encoder_forward(source_embeddings,source_padding_mask,output_all_layers)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs,source_padding_mask,target,output_all_layers)

        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = source_embeddings
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, model_all_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).to(device).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def calculate_loss(self, batch_data:dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'equation'.
        """
        src = torch.tensor(batch_data['question']).to(self.device)
        target = torch.tensor(batch_data['equation']).to(self.device)
        token_logits, _, _ = self.forward(src, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data:dict) -> tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'equation' and 'num list'.
        """
        src = torch.tensor(batch_data['question']).to(self.device)
        target = torch.tensor(batch_data['equation']).to(self.device)
        num_list = batch_data['num list']

        _, symbol_outputs, _ = self.forward(src)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self,batch_data:dict,output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question']).to(self.device)
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq,output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self,seq_emb,seq_mask,output_all_layers=False):
        encoder_outputs = self.encoder(seq_emb, self_padding_mask=seq_mask,output_all_encoded_layers=output_all_layers)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs']=encoder_outputs
            return encoder_outputs[-1],all_layer_outputs
        return encoder_outputs,all_layer_outputs

    def decoder_forward(self,encoder_outputs,seq_mask,target=None,output_all_layers=False):
        with_t = random.random()
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if target is not None and with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, -1).to(device)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]

            decoder_inputs = self.out_embedder(target) + self.pos_embedder(target).to(device)
            self_padding_mask = torch.eq(target, self.out_pad_token)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()

            decoder_outputs = self.decoder(decoder_inputs,
                                           self_padding_mask=self_padding_mask,
                                           self_attn_mask=self_attn_mask,
                                           external_states=encoder_outputs,
                                           external_padding_mask=seq_mask)
            token_logits = self.out(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            token_logits = []
            outputs = []
            seq_len = target.size(1) if target is not None else self.max_output_len
            input_seq = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, -1).to(device)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq).to(device)
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask,
                                               external_states=encoder_outputs,
                                               external_padding_mask=seq_mask)

                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                # output=greedy_search(token_logit)
                output = torch.topk(token_logit.squeeze(), 1, dim=-1)[1]
                outputs.append(output)
                if self.share_vocab:
                    pre_tokens.append(self.convert_out_idx_2_in_idx(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            outputs = torch.stack(outputs,dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

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


