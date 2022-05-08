# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 05:00:30
# @File: trnn.py

import copy
import random

from torch.nn.functional import cross_entropy
from typing import Tuple

from mwptoolkit.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from mwptoolkit.loss.nll_loss import NLLLoss
# from mwptoolkit.loss.cross_entropy_loss import CrossEntropyLoss
import torch
from torch import nn

from mwptoolkit.module.Layer.tree_layers import RecursiveNN
from mwptoolkit.module.Encoder.rnn_encoder import SelfAttentionRNNEncoder, BasicRNNEncoder
from mwptoolkit.module.Attention.seq_attention import SeqAttention
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Embedder.roberta_embedder import RobertaEmbedder
from mwptoolkit.module.Embedder.bert_embedder import BertEmbedder
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec
from mwptoolkit.utils.data_structure import Node, BinaryTree
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class TRNN(nn.Module):
    """
    Reference:
        Wang et al. "Template-Based Math Word Problem Solvers with Recursive Neural Networks" in AAAI 2019.
    """

    def __init__(self, config, dataset):
        super(TRNN, self).__init__()
        self.device = config['device']
        self.seq2seq_embedding_size = config["seq2seq_embedding_size"]
        self.seq2seq_encode_hidden_size = config["seq2seq_encode_hidden_size"]
        self.seq2seq_decode_hidden_size = config["seq2seq_decode_hidden_size"]
        self.num_layers = config["seq2seq_num_layers"]
        self.teacher_force_ratio = config["teacher_force_ratio"]
        self.seq2seq_dropout_ratio = config['seq2seq_dropout_ratio']
        self.ans_embedding_size = config["ans_embedding_size"]
        self.ans_hidden_size = config["ans_hidden_size"]
        self.ans_dropout_ratio = config["ans_dropout_ratio"]
        self.ans_num_layers = config["ans_num_layers"]

        self.encoder_rnn_cell_type = config["encoder_rnn_cell_type"]
        self.decoder_rnn_cell_type = config["decoder_rnn_cell_type"]
        self.max_gen_len = config["max_output_len"]
        self.bidirectional = config["bidirectional"]
        self.attention = True
        self.share_vocab = config["share_vocab"]
        self.embedding = config["embedding"]

        self.mask_list = NumMask.number
        self.in_idx2word = dataset.in_idx2word
        self.out_idx2symbol = dataset.out_idx2symbol
        self.temp_idx2symbol = dataset.temp_idx2symbol
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.temp_symbol_size = len(dataset.temp_idx2symbol)
        self.operator_nums = len(dataset.operator_list)
        self.operator_list = dataset.operator_list
        self.generate_list = [SpecialTokens.UNK_TOKEN] + dataset.generate_list
        self.generate_idx = [self.in_idx2word.index(num) for num in self.generate_list]

        if self.share_vocab:
            self.sos_token_idx = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.sos_token_idx = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        try:
            self.out_sos_token = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None

        try:
            self.temp_sos_token = dataset.temp_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.temp_sos_token = None
        try:
            self.temp_eos_token = dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.temp_eos_token = None
        try:
            self.temp_pad_token = dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.temp_pad_token = None

        # seq2seq module
        if config['embedding'] == 'roberta':
            self.seq2seq_in_embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.seq2seq_in_embedder.token_resize(self.vocab_size)
        elif config['embedding'] == 'bert':
            self.seq2seq_in_embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.seq2seq_in_embedder.token_resize(self.vocab_size)
        else:
            self.seq2seq_in_embedder = BasicEmbedder(self.vocab_size, self.seq2seq_embedding_size,
                                                     self.seq2seq_dropout_ratio)
        if self.share_vocab:
            self.seq2seq_out_embedder = self.seq2seq_in_embedder
        else:
            self.seq2seq_out_embedder = BasicEmbedder(self.temp_symbol_size, self.seq2seq_embedding_size,
                                                      self.seq2seq_dropout_ratio)
        self.seq2seq_encoder = BasicRNNEncoder(self.seq2seq_embedding_size, self.seq2seq_encode_hidden_size,
                                               self.num_layers, \
                                               self.encoder_rnn_cell_type, self.seq2seq_dropout_ratio,
                                               self.bidirectional)
        self.seq2seq_decoder = AttentionalRNNDecoder(self.seq2seq_embedding_size, self.seq2seq_decode_hidden_size,
                                                     self.seq2seq_encode_hidden_size, \
                                                     self.num_layers, self.decoder_rnn_cell_type,
                                                     self.seq2seq_dropout_ratio)
        self.seq2seq_gen_linear = nn.Linear(self.seq2seq_encode_hidden_size, self.temp_symbol_size)
        # answer module
        if config['embedding'] == 'roberta':
            self.answer_in_embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.answer_in_embedder.token_resize(self.vocab_size)
        elif config['embedding'] == 'bert':
            self.answer_in_embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.answer_in_embedder.token_resize(self.vocab_size)
        else:
            self.answer_in_embedder = BasicEmbedder(self.vocab_size, self.ans_embedding_size, self.ans_dropout_ratio)
        self.answer_encoder = SelfAttentionRNNEncoder(self.ans_embedding_size, self.ans_hidden_size,
                                                      self.ans_embedding_size, self.num_layers, \
                                                      self.encoder_rnn_cell_type, self.ans_dropout_ratio,
                                                      self.bidirectional)
        self.answer_rnn = RecursiveNN(self.ans_embedding_size, self.operator_nums, self.operator_list)

        weight = torch.ones(self.temp_symbol_size).to(config["device"])
        pad = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.seq2seq_loss = NLLLoss(weight, pad)
        weight2 = torch.ones(self.operator_nums).to(config["device"])
        self.ans_module_loss = NLLLoss(weight2, size_average=True)
        # self.ans_module_loss=CrossEntropyLoss(weight2,size_average=True)

        self.wrong = 0

    def forward(self, seq, seq_length, seq_mask, num_pos, template_target=None, equation_target=None,
                output_all_layers=False):
        seq2seq_token_logits, seq2seq_outputs, seq2seq_layer_outputs = self.seq2seq_forward(seq, seq_length,
                                                                                            template_target,
                                                                                            output_all_layers)
        if equation_target:
            template = None
        else:
            template = self.convert_temp_idx2symbol(seq2seq_outputs)
        ans_token_logits, ans_outputs, ans_module_layer_outputs = self.ans_module_forward(seq, seq_length, seq_mask,
                                                                                          template, num_pos,
                                                                                          equation_target,
                                                                                          output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(seq2seq_layer_outputs)
            model_all_outputs.update(ans_module_layer_outputs)

        return (seq2seq_token_logits, ans_token_logits), (seq2seq_outputs, ans_outputs), model_all_outputs

    def calculate_loss(self, batch_data: dict) -> Tuple[float, float]:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: seq2seq module loss, answer module loss.
        """

        # first stage:train seq2seq
        seq2seq_loss = self.seq2seq_calculate_loss(batch_data)

        # second stage: train answer module
        answer_loss = self.ans_module_calculate_loss(batch_data)

        return seq2seq_loss, answer_loss

    def model_test(self, batch_data: dict) -> tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation', 'ques mask',
        'num pos', 'num list', 'template'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)

        seq_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)
        num_pos = batch_data['num pos']
        num_list = batch_data["num list"]
        template_target = self.convert_temp_idx2symbol(torch.tensor(batch_data['template']))

        _, output_template, _ = self.seq2seq_forward(seq, seq_length)
        template = self.convert_temp_idx2symbol(output_template)

        _, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template, num_pos,
                                                          output_all_layers=True)
        equations = ans_module_layers['ans_model_equation_outputs']
        _, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template_target, num_pos,
                                                          output_all_layers=True)
        ans_module_test = ans_module_layers['ans_model_equation_outputs']

        equations = self.mask2num(equations, num_list)
        ans_module_test = self.mask2num(ans_module_test, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        temp_t = template_target
        return equations, targets, template, temp_t, ans_module_test, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        ques_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, ques_mask, num_pos,
                                                                       output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def seq2seq_calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation of seq2seq module.

        :param batch_data: one batch data.
        :return: loss value of seq2seq module.
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["template"]).to(self.device)
        # ques_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)

        token_logits, _, _ = self.seq2seq_forward(seq, seq_length, target)

        if self.share_vocab:
            target = self.convert_in_idx_2_temp_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.seq2seq_loss.reset()
        self.seq2seq_loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.seq2seq_loss.backward()
        return self.seq2seq_loss.get_loss()

    def ans_module_calculate_loss(self, batch_data):
        """Finish forward-propagating, calculating loss and back-propagation of answer module.

        :param batch_data: one batch data.
        :return: loss value of answer module.
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        seq_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)

        num_pos = batch_data["num pos"]
        equ_source = copy.deepcopy(batch_data["equ_source"])

        for idx, equ in enumerate(equ_source):
            equ_source[idx] = equ.split(" ")
        template = equ_source

        token_logits, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template, num_pos,
                                                                     equation_target=template, output_all_layers=True)
        target = ans_module_layers["ans_module_target"]

        self.ans_module_loss.reset()
        for b_i in range(len(target)):
            if not isinstance(token_logits[b_i],list):
                output = torch.nn.functional.log_softmax(token_logits[b_i], dim=1)
                self.ans_module_loss.eval_batch(output, target[b_i].view(-1))
        self.ans_module_loss.backward()
        return self.ans_module_loss.get_loss()

    def seq2seq_generate_t(self, encoder_outputs, encoder_hidden, decoder_inputs):
        with_t = random.random()
        if with_t < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden)
            token_logits = self.seq2seq_gen_linear(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
            token_logits = torch.nn.functional.log_softmax(token_logits, dim=1)
        else:
            seq_len = decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            token_logits = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
                # attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                token_logit = self.seq2seq_gen_linear(step_output)
                predict = torch.nn.functional.log_softmax(token_logit, dim=1)
                # predict=torch.log_softmax(token_logit,dim=1)
                output = predict.topk(1, dim=1)[1]
                token_logits.append(predict)

                if self.share_vocab:
                    output = self.convert_temp_idx_2_in_idx(output)
                    decoder_input = self.seq2seq_out_embedder(output)
                else:
                    decoder_input = self.seq2seq_out_embedder(output)
            token_logits = torch.stack(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def seq2seq_generate_without_t(self, encoder_outputs, encoder_hidden, decoder_input):
        all_outputs = []
        decoder_hidden = encoder_hidden
        for idx in range(self.max_gen_len):
            if self.attention:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logits = self.seq2seq_gen_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logits, dim=1)
            output = predict.topk(1, dim=1)[1]

            all_outputs.append(output)
            if self.share_vocab:
                output = self.convert_temp_idx_2_in_idx(output)
                decoder_input = self.seq2seq_out_embedder(output)
            else:
                decoder_input = self.seq2seq_out_embedder(output)
        all_outputs = torch.cat(all_outputs, dim=1)
        return all_outputs

    def seq2seq_forward(self, seq, seq_length, target=None, output_all_layers=False):
        batch_size = seq.size(0)
        device = seq.device

        seq_emb = self.seq2seq_in_embedder(seq)

        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.seq2seq_encoder_forward(seq_emb, seq_length,
                                                                                              output_all_layers)

        decoder_inputs = self.init_seq2seq_decoder_inputs(target, device, batch_size)

        token_logits, symbol_outputs, decoder_layer_outputs = self.seq2seq_decoder_forward(encoder_outputs,
                                                                                           encoder_hidden,
                                                                                           decoder_inputs, target,
                                                                                           output_all_layers)

        seq2seq_all_outputs = {}
        if output_all_layers:
            seq2seq_all_outputs['seq2seq_inputs_embedding'] = seq_emb
            seq2seq_all_outputs.update(encoder_layer_outputs)
            seq2seq_all_outputs.update(decoder_layer_outputs)

        return token_logits, symbol_outputs, seq2seq_all_outputs

    def ans_module_forward(self, seq, seq_length, seq_mask, template, num_pos, equation_target=None,
                           output_all_layers=False):
        if self.embedding == 'roberta':
            seq_emb = self.answer_in_embedder(seq, seq_mask)
        else:
            seq_emb = self.answer_in_embedder(seq)
        encoder_output, encoder_hidden = self.answer_encoder(seq_emb, seq_length)
        batch_size = encoder_output.size(0)
        generate_num = torch.tensor(self.generate_idx).to(self.device)
        if self.embedding == 'roberta':
            generate_emb = self.answer_in_embedder(generate_num, None)
        else:
            generate_emb = self.answer_in_embedder(generate_num)

        batch_prob = []
        batch_target = []
        outputs = []
        equations = []
        input_template = equation_target if equation_target else template
        if equation_target is not None:
            for b_i in range(batch_size):
                try:
                    tree_i = self.template2tree(input_template[b_i])
                except IndexError:
                    outputs.append([])
                    continue
                look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
                num_encoding = seq_emb[b_i, num_pos[b_i]] + encoder_output[b_i, num_pos[b_i]]
                num_embedding = torch.cat([generate_emb, num_encoding], dim=0)
                assert len(look_up) == len(num_embedding)
                prob, target = self.answer_rnn(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
                batch_prob.append(prob)
                batch_target.append(target)
                if not isinstance(prob,list):
                    output = torch.topk(prob, 1)[1]
                    outputs.append(output)
                else:
                    outputs.append([])
        else:
            for b_i in range(batch_size):
                try:
                    tree_i = self.template2tree(input_template[b_i])
                except IndexError:
                    outputs.append([])
                    continue
                look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
                num_encoding = seq_emb[b_i, num_pos[b_i]] + encoder_output[b_i, num_pos[b_i]]
                num_embedding = torch.cat([generate_emb, num_encoding], dim=0)
                assert len(look_up) == len(num_embedding)
                prob, output, node_pred = self.answer_rnn.test(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
                batch_prob.append(prob)
                tree_i.root = node_pred
                outputs.append(output)
                equation = self.tree2equation(tree_i)
                equations.append(equation)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['ans_module_token_logits'] = batch_prob
            all_layer_outputs['ans_module_target'] = batch_target
            all_layer_outputs['ans_model_outputs'] = outputs
            all_layer_outputs['ans_model_equation_outputs'] = equations
        return batch_prob, outputs, all_layer_outputs

    def seq2seq_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.seq2seq_encoder(seq_emb, seq_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.seq2seq_encode_hidden_size:] + encoder_outputs[:, :,
                                                                                        :self.seq2seq_encode_hidden_size]
            if self.encoder_rnn_cell_type == 'lstm':
                encoder_hidden = (encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous())
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder_rnn_cell_type == self.decoder_rnn_cell_type:
            pass
        elif (self.encoder_rnn_cell_type == 'gru') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'rnn') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'lstm') and (
                self.decoder_rnn_cell_type == 'gru' or self.decoder_rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['seq2seq_encoder_outputs'] = encoder_outputs
            all_layer_outputs['seq2seq_encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def seq2seq_decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None,
                                output_all_layers=False):
        if target is not None and random.random() < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden)
            token_logits = self.seq2seq_gen_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.seq2seq_gen_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)

                if self.share_vocab:
                    output = self.convert_temp_idx_2_in_idx(output)
                    decoder_input = self.seq2seq_out_embedder(output)
                else:
                    decoder_input = self.seq2seq_out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['seq2seq_decoder_outputs'] = decoder_outputs
            all_layer_outputs['seq2seq_token_logits'] = token_logits
            all_layer_outputs['seq2seq_outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def template2tree(self, template):
        tree = BinaryTree()
        tree.equ2tree_(template)
        return tree

    def tree2equation(self, tree):
        equation = tree.tree2equ(tree.root)
        return equation

    def init_seq2seq_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).to(device).view(batch_size, 1)
        if target is not None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.seq2seq_out_embedder(decoder_inputs)
        return decoder_inputs

    def convert_temp_idx_2_in_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.temp_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_temp_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.temp_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_temp_idx2symbol(self, output):
        batch_size = output.size(0)
        seq_len = output.size(1)
        symbol_list = []
        for b_i in range(batch_size):
            symbols = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.temp_sos_token, self.temp_eos_token, self.temp_pad_token]:
                    break
                symbol = self.temp_idx2symbol[idx]
                symbols.append(symbol)
            symbol_list.append(symbols)
        return symbol_list

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

    def symbol2idx(self, symbols):
        r"""symbol to idx
        equation symbol to equation idx
        """
        outputs = []
        for symbol in symbols:
            if symbol not in self.out_idx2symbol:
                idx = self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)
            else:
                idx = self.out_idx2symbol.index(symbol)
            outputs.append(idx)
        return outputs

    def mask2num(self, output, num_list):
        batch_size = len(output)
        output_list = []
        for b_i in range(batch_size):
            res = []
            seq_len = len(output[b_i])
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                symbol = output[b_i][s_i]
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
