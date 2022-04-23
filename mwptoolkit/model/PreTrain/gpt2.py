# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:36:11
# @File: gpt2.py
from typing import Tuple, Dict, Any

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer, GPT2Tokenizer

from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask, DatasetName


class GPT2(nn.Module):
    """
    Reference:
        Radford et al. "Language Models are Unsupervised Multitask Learners".
    """

    def __init__(self, config, dataset):
        super(GPT2, self).__init__()
        self.device = config["device"]
        self.max_out_len = config['max_output_len']
        self.max_input_len = config["max_len"]

        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config[
            'transformers_pretrained_model']

        self.tokenizer = dataset.tokenizer
        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            # print ("tokenizer: ")
            self.eos_token_id = self.tokenizer.sep_token_id
            self.eos_token = self.tokenizer.sep_token
            self.start_token = self.tokenizer.cls_token
        else:
            self.eos_token_id = self.tokenizer.eos_token_id
            self.eos_token = self.tokenizer.eos_token
            self.start_token = ''

        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self._pretrained_model_resize()

        self.loss = NLLLoss()

    def _pretrained_model_resize(self):
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None,output_all_layers=False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor | None target: target, shape: [batch_size,target_length].
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits: [batch_size, output_length, output_size], symbol_outputs: [batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(seq, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data:dict)->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        Args:
            batch_data (dict): one batch data.

        Returns:
            float: loss value.
        """
        seq, target = batch_data["question"], batch_data["equation"]
        seq = torch.LongTensor(seq).to(self.device)
        target = torch.LongTensor(target).to(self.device)
        token_logits,_,_ = self.forward(seq, target)
        token_logits = token_logits.view(-1,token_logits.size(-1))
        outputs = torch.nn.functional.log_softmax(token_logits, dim=1)

        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()

        return self.loss.get_loss()

    def model_test(self, batch_data:dict)->tuple:
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
        targets = self.convert_idx2symbol(target, num_list)
        return outputs, targets

    def predict(self, batch_data:dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question']).to(self.device)
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def list2str(self, x):
        y = ''.join(x)
        return y

    def decoder_forward(self,seq,target=None,output_all_layers=False):
        if target is not None:
            tgts_inputs_tensor = target[:, :-1]  # '[CLS] / * num_1 num_2 num_0 [SEP]
            tgts_outputs_tensor = target  # '[CLS] / * num_1 num_2 num_0 [SEP] [SEP]'
            seq_mask = (tgts_inputs_tensor != self.eos_token_id).float()
            seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.), seq_mask], 1)

            inputs = torch.cat([seq, tgts_inputs_tensor], 1)
            logits = self.decoder(inputs)[0]
            logits = logits[:, -tgts_outputs_tensor.shape[1]:, :].contiguous()
            outputs = torch.topk(logits,1,dim=-1)[1]
        else:
            outputs = []
            logits = []
            inputs = seq
            for idx in range(self.max_out_len):
                decoder_outputs = self.decoder(inputs)
                token_logit = decoder_outputs[0][:, -1, :]
                tokens = token_logit.topk(1, dim=1)[1]
                # mask=tokens==self.tokenizer.pad_token_id
                logits.append(token_logit)
                outputs.append(tokens)
                inputs = torch.cat((inputs, tokens), dim=1)
            logits = torch.stack(logits,dim=1)
            outputs = torch.cat(outputs, dim=1)
            # all_output = self.decode_(all_output)
            # print (all_output)
            # print ("all_output:", all_output.size())
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits']=logits
            all_layer_outputs['outputs']=outputs
        return logits,outputs,all_layer_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = self.tokenizer.decode(outputs[b])
            symbols = self.tokenizer.tokenize(symbols)
            symbols_ = []
            for token in symbols:
                if token == self.start_token:
                    continue
                if 'Ġ' in token:
                    symbols_.append(token[1:])
                # if '/' == token[0] and len(token) == 2 and ('+' == token[1] or  '-' == token[1] or '*' == token[1] or '/' == token[1]):
                #     symbols_.append(token[0])
                #     symbols_.append(token[1:])
                elif token == self.eos_token:
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            # print ("symbols",symbols)
            all_outputs.append(symbols)
        # print (all_outputs)
        return all_outputs

    def encode_(self, inputs):
        outputs = []
        for idx, s in enumerate(inputs):
            out = self.tokenizer.encode(inputs[idx])
            outputs.append(out)

        output_length = max([len(_) for _ in outputs]) + 1
        for i in range(len(outputs)):
            outputs[i] += (output_length - len(outputs[i])) * [self.eos_token_id]
        outputs_tensor = torch.LongTensor(outputs)

        return outputs_tensor

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


