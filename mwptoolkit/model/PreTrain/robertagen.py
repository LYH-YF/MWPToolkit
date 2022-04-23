# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:37:24
# @File: robertagen.py

import random
from typing import Tuple, Dict, Any

import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer

from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder
from mwptoolkit.module.Embedder.position_embedder import PositionEmbedder_x as PositionEmbedder
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Attention.self_attention import SelfAttentionMask
from mwptoolkit.module.Strategy.sampling import topk_sampling
from mwptoolkit.module.Strategy.greedy import greedy_search
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask, DatasetName


class RobertaGen(nn.Module):
    """
    Reference:
        Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
    """

    def __init__(self, config, dataset):
        super(RobertaGen, self).__init__()
        self.device = config["device"]
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config[
            'transformers_pretrained_model']
        self.max_input_len = config['max_len']
        self.max_output_len = config['max_output_len']

        self.tokenizer = dataset.tokenizer
        self.eos_token_id = self.tokenizer.sep_token_id
        self.eos_token = self.tokenizer.sep_token
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol

        self.max_output_len = config["max_output_len"]
        self.share_vocab = config["share_vocab"]
        self.decoding_strategy = config["decoding_strategy"]
        self.teacher_force_ratio = config['teacher_force_ratio']

        self.out_pad_idx = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.out_sos_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        self.out_eos_idx = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        self.out_unk_idx = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]

        config["vocab_size"] = len(self.tokenizer)
        config["symbol_size"] = len(self.out_symbol2idx)
        config["in_word2idx"] = self.tokenizer.get_vocab()
        config["in_idx2word"] = list(self.tokenizer.get_vocab().keys())
        # config["embedding_size"] = self.encoder.config.n_embd
        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        else:
            self.encoder = RobertaModel.from_pretrained(self.pretrained_model_path)

        self.in_embedder = BasicEmbedder(config["vocab_size"], config["embedding_size"],
                                         config["embedding_dropout_ratio"])

        self.out_embedder = BasicEmbedder(config["symbol_size"], config["embedding_size"],
                                          config["embedding_dropout_ratio"])

        self.pos_embedder = PositionEmbedder(config["embedding_size"], config["max_len"])
        self.self_attentioner = SelfAttentionMask()

        self.decoder = TransformerDecoder(config["embedding_size"], config["ffn_size"], config["num_decoder_layers"], \
                                          config["num_heads"], config["attn_dropout_ratio"], \
                                          config["attn_weight_dropout_ratio"], config["ffn_dropout_ratio"])
        self.out = nn.Linear(config["embedding_size"], config["symbol_size"])

        self.loss = NLLLoss()
        self._pretrained_model_resize()

    def _pretrained_model_resize(self):
        self.encoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None, output_all_layers=False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor | None target: target, shape: [batch_size,target_length].
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits: [batch_size, output_length, output_size], symbol_outputs: [batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_feat, encoder_layer_outputs = self.encoder_forward(seq, output_all_layers)

        source_padding_mask = torch.eq(seq, self.tokenizer.pad_token_id)

        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(seq_feat, source_padding_mask,
                                                                                   target, output_all_layers)

        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.

        Args:
            batch_data (dict): one batch data.

        Returns:
            float: loss value.
        """
        seq, target = batch_data["question"], batch_data["equation"]
        seq = torch.LongTensor(seq).to(self.device)
        target = torch.LongTensor(target).to(self.device)

        token_logits, _, _ = self.forward(seq, target)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        outputs = torch.nn.functional.log_softmax(token_logits, dim=1)

        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()

        return self.loss.get_loss()

    def model_test(self, batch_data: dict) -> tuple:
        """Model test.

        Args:
            batch_data (dict): one batch data.

        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data["question"]

        num_list = batch_data['num list']
        target = batch_data['equation']

        seq = torch.LongTensor(seq).to(self.device)
        target = torch.LongTensor(target).to(self.device)
        _, outputs, _ = self.forward(seq)

        outputs = self.decode_(outputs)
        target = self.decode_(target)
        outputs = self.convert_idx2symbol(outputs, num_list)
        target = self.convert_idx2symbol(target, num_list)
        return outputs, target

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question']).to(self.device)
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq, output_all_layers=False):
        encoder_outputs = self.encoder(seq, return_dict=True)
        src_feat = encoder_outputs[0]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
        return src_feat, all_layer_outputs

    def decoder_forward(self, encoder_outputs, source_padding_mask, target=None, output_all_layers=None):
        with_t = random.random()
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if target is not None and with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]

            decoder_inputs = self.pos_embedder(self.out_embedder(target))
            self_padding_mask = torch.eq(target, self.out_pad_idx)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            decoder_outputs = self.decoder(decoder_inputs,
                                           self_padding_mask=self_padding_mask,
                                           self_attn_mask=self_attn_mask,
                                           external_states=encoder_outputs,
                                           external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            outputs = torch.topk(token_logits, 1, dim=-1)[1].squeeze(-1)
            # token_logits = token_logits.view(-1, token_logits.size(-1))
        else:
            token_logits = []
            outputs = []
            seq_len = target.size(1) if target is not None else self.max_output_len
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask,
                                               external_states=encoder_outputs,
                                               external_padding_mask=source_padding_mask)

                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                if self.decoding_strategy == "topk_sampling":
                    output = topk_sampling(token_logit, top_k=5)
                elif self.decoding_strategy == "greedy_search":
                    output = greedy_search(token_logit)
                else:
                    raise NotImplementedError
                outputs.append(output)
                if self.share_vocab:
                    pre_tokens.append(self.decode(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            # token_logits = token_logits.view(-1, token_logits.size(-1))
            outputs = torch.cat(outputs, dim=1)

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = [self.out_idx2symbol[_] for _ in outputs[b]]
            symbols_ = []
            for token in symbols:
                # if '/' == token[0] and len(token) == 2 and (
                #         '+' == token[1] or '-' == token[1] or '*' == token[1] or '/' == token[1]):
                #     symbols_.append(token[0])
                #     symbols_.append(token[1:])
                if token == SpecialTokens.EOS_TOKEN or token == SpecialTokens.PAD_TOKEN:
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            # print ("symbols",symbols)
            all_outputs.append(symbols)
        # print (all_outputs)
        return all_outputs

    def decode(self, output):
        device = output.device

        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return output

    def convert_idx2symbol(self, outputs, num_lists):
        batch_size = len(outputs)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_lists[b_i])
            res = []
            if isinstance(outputs[b_i], str):
                output = outputs[b_i].split()
            else:
                output = outputs[b_i]
            for s_i in range(len(output)):
                symbol = output[s_i]
                if "NUM" in symbol:
                    num_idx = NumMask.number.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_lists[b_i][num_idx])
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
