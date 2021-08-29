# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:36:11
# @File: gpt2.py

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

        #self.eval_generate_num = config['eval_generate_num']
        self.device = config["device"]
        self.max_out_len = config['max_output_len']
        self.max_input_len = config["max_len"]

        self.pretrained_model_path = config['pretrained_model_path']

        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            # print ("tokenizer: ")
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
            self.eos_token_id = self.tokenizer.sep_token_id
            self.eos_token = self.tokenizer.sep_token
            self.start_token = self.tokenizer.cls_token
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_path)
            self.eos_token_id = self.tokenizer.eos_token_id
            self.eos_token = self.tokenizer.eos_token
            self.start_token = ''

        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self.init_tokenizer_and_resize(dataset.generate_list, NumMask.number[:dataset.copy_nums], dataset.operator_list)
        #self.padding_token_idx = self.tokenizer.pad_token_id

        # config["vocab_size"] = len(self.tokenizer)
        # config["symbol_size"] = len(self.tokenizer)
        # config["embedding_size"] = len(self.tokenizer)
        # config["in_word2idx"] = self.tokenizer.get_vocab()
        # config["in_idx2word"] = list(self.tokenizer.get_vocab().keys())
        # config["out_symbol2idx"] = self.tokenizer.get_vocab()
        # config["out_idx2symbol"] = list(self.tokenizer.get_vocab().keys())

        self.loss = NLLLoss()

    def init_tokenizer_and_resize(self, generate_list, mask_number_list, operator_list):
        _ = self.tokenizer.add_tokens(operator_list)
        _ = self.tokenizer.add_tokens(generate_list)
        _ = self.tokenizer.add_tokens(mask_number_list)
        #self.tokenizer.add_special_tokens({"eos_token":SpecialTokens.EOS_TOKEN})
        #self.tokenizer.eos_token=self.tokenizer.sep_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ans>"]}, )
        SpecialTokens.PAD_TOKEN = self.tokenizer.eos_token
        #SpecialTokens.EOS_TOKEN=self.eos_token
        SpecialTokens.UNK_TOKEN = self.tokenizer.unk_token
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None):

        if target != None:
            token_logits, target = self.generate_t(seq, target)
            return token_logits, target
        else:
            all_output = self.generate_without_t(seq)
            return all_output, None

    def calculate_loss(self, batch_data):
        """Finish forward-propagating, calculating loss and back-propagation.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            float: loss value.
        """
        seq, target = batch_data["ques_source"], batch_data["equ_source"]
        outputs, target = self.forward(seq, target)
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)

        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()

        return self.loss.get_loss()

    def model_test(self, batch_data):
        """Model test.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data["ques_source"]

        num_list = batch_data['num list']
        target = batch_data['equ_source']

        outputs = self.generate_without_t(seq)
        batch_size = len(target)

        outputs = self.convert_idx2symbol(outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return outputs, targets

    def list2str(self, x):
        y = ''.join(x)
        return y

    def generate_t(self, seq, target=None):
        srcs = []
        tgts = []
        for idx, s in enumerate(seq):
            src = self.tokenizer.encode(seq[idx])
            tgt = self.tokenizer.encode(target[idx])
            srcs.append(src)
            tgts.append(tgt)

        if self.max_input_len is not None:
            src_length = self.max_input_len - 1
        else:
            src_length = max([len(_) for _ in srcs]) + 1
        tgt_length = max([len(_) for _ in tgts]) + 1

        for i in range(len(tgts)):
            tgts[i] += (tgt_length - len(tgts[i])) * [self.eos_token_id]
        tgts_tensor = torch.LongTensor(tgts)

        for i in range(len(srcs)):
            if src_length >= len(srcs[i]):
                srcs[i] = (src_length - len(srcs[i])) * [self.eos_token_id] + srcs[i] + self.tokenizer.encode(['<ans>'])
            else:
                srcs[i] = srcs[i][:src_length] + self.tokenizer.encode(['<ans>'])
        srcs_tensor = torch.LongTensor(srcs)
        src_length += 1

        seq_mask = (tgts_tensor != self.eos_token_id)[:, :-1].float()
        seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.), seq_mask], 1)

        tgts_inputs_tensor = tgts_tensor[:, :-1]  #'[CLS] / * num_1 num_2 num_0 [SEP]
        tgts_outputs_tensor = tgts_tensor  #'[CLS] / * num_1 num_2 num_0 [SEP] [SEP]'

        srcs_tensor = srcs_tensor.to(self.device)
        tgts_tensor = tgts_tensor.to(self.device)
        tgts_inputs_tensor = tgts_inputs_tensor.to(self.device)
        tgts_outputs_tensor = tgts_outputs_tensor.to(self.device)
        seq_mask = seq_mask.to(self.device)

        inputs = torch.cat([srcs_tensor, tgts_inputs_tensor], 1)
        logits = self.decoder(inputs)[0]
        logits = logits[:, -tgts_outputs_tensor.shape[1]:, :].contiguous()
        logits = logits.view(-1, logits.shape[-1])
        return logits, tgts_outputs_tensor

    def generate_without_t(self, seq):

        srcs = []
        for idx, s in enumerate(seq):
            src = self.tokenizer.encode(seq[idx])
            srcs.append(src)
        if self.max_input_len is not None:
            src_length = self.max_input_len - 1
        else:
            src_length = max([len(_) for _ in srcs]) + 1

        for i in range(len(srcs)):
            if src_length >= len(srcs[i]):
                srcs[i] = (src_length - len(srcs[i])) * [self.eos_token_id] + srcs[i] + self.tokenizer.encode(['<ans>'])
            else:
                srcs[i] = srcs[i][:src_length] + self.tokenizer.encode(['<ans>'])
        srcs_tensor = torch.LongTensor(srcs)
        src_length += 1

        srcs_tensor = srcs_tensor.to(self.device)
        inputs = srcs_tensor

        all_output = []
        for idx in range(self.max_out_len):
            outputs = self.decoder(inputs)
            token_logit = outputs[0][:, -1, :]
            tokens = token_logit.topk(1, dim=1)[1]
            # mask=tokens==self.tokenizer.pad_token_id
            all_output.append(tokens)
            inputs = torch.cat((inputs, tokens), dim=1)
        all_output = torch.cat(all_output, dim=1)
        all_output = self.decode_(all_output)
        # print (all_output)
        # print ("all_output:", all_output.size())
        return all_output

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


# class GPT2(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         #self.eval_generate_num = config['eval_generate_num']
#         self.device = config["device"]
#         self.pretrained_model_path = config['pretrained_model_path']

#         #self.tokenizer=GPT2Tokenizer.from_pretrained(self.pretrained_model_path)
#         self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)

#         self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

#         self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)

#         self.init_tokenizer_and_resize(config["generate_list"],NumMask.number,config["operator_list"])
#         self.padding_token_idx = self.tokenizer.pad_token_id
#         self.max_out_len = config['max_output_len']
#     def init_tokenizer_and_resize(self,generate_list,mask_number_list,operator_list):
#         _ = self.tokenizer.add_tokens(operator_list)
#         _ = self.tokenizer.add_tokens(generate_list)
#         _ = self.tokenizer.add_tokens(mask_number_list)
#         #self.tokenizer.add_special_tokens({"eos_token":SpecialTokens.EOS_TOKEN})
#         self.tokenizer.eos_token=self.tokenizer.sep_token
#         #self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ans>"]},)
#         SpecialTokens.PAD_TOKEN=self.tokenizer.pad_token
#         SpecialTokens.EOS_TOKEN=self.tokenizer.eos_token
#         SpecialTokens.UNK_TOKEN=self.tokenizer.unk_token
#         self.decoder.resize_token_embeddings(len(self.tokenizer))
#     def forward(self, seq, target=None):

#         if target != None:
#             token_logits, target = self.generate_t(seq, target)
#             return token_logits, target
#         else:
#             all_output, target = self.generate_without_t(seq)
#             return all_output, target

#     def list2str(self, x):
#         y = ''.join(x)
#         return y

#     def generate_t(self, seq, target=None):
#         srcs = []
#         tgts = []
#         for idx, s in enumerate(seq):
#             src = self.tokenizer.encode(seq[idx])
#             tgt = self.tokenizer.encode(target[idx])
#             srcs.append(src)
#             tgts.append(tgt)

#         src_length = max([len(_) for _ in srcs]) + 1
#         tgt_length = max([len(_) for _ in tgts]) + 1

#         for i in range(len(tgts)):
#             tgts[i] += (tgt_length - len(tgts[i])) * [self.tokenizer.eos_token_id]
#         tgts_tensor = torch.LongTensor(tgts)

#         for i in range(len(srcs)):
#             srcs[i] = (src_length - len(srcs[i])) * [self.tokenizer.eos_token_id] + srcs[i] + self.tokenizer.encode(['<ans>'])
#         srcs_tensor = torch.LongTensor(srcs)
#         src_length += 1

#         seq_mask = (tgts_tensor != self.tokenizer.eos_token_id)[:, :-1].float()
#         seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.), seq_mask], 1)

#         tgts_inputs_tensor = tgts_tensor[:, :-1]
#         tgts_outputs_tensor = tgts_tensor

#         srcs_tensor = srcs_tensor.to(self.device)
#         tgts_tensor = tgts_tensor.to(self.device)
#         tgts_inputs_tensor = tgts_inputs_tensor.to(self.device)
#         tgts_outputs_tensor = tgts_outputs_tensor.to(self.device)
#         seq_mask = seq_mask.to(self.device)

#         inputs = torch.cat([srcs_tensor, tgts_inputs_tensor], 1)
#         logits = self.decoder(inputs)[0]
#         logits = logits[:, -tgts_outputs_tensor.shape[1]:, :].contiguous()
#         logits = logits.view(-1, logits.shape[-1])
#         return logits, tgts_outputs_tensor

#     def generate_without_t(self, seq, target=None):
#         srcs = []
#         tgts = []

#         for idx, s in enumerate(seq):
#             src = self.tokenizer.encode(seq[idx])
#             tgt = self.tokenizer.encode(target[idx])
#             srcs.append(src)
#             tgts.append(tgt)

#         src_length = max([len(_) for _ in srcs]) + 1
#         tgt_length = max([len(_) for _ in tgts]) + 1

#         for i in range(len(tgts)):
#             tgts[i] += (tgt_length - len(tgts[i])) * [self.tokenizer.eos_token_id]
#         tgts_tensor = torch.LongTensor(tgts)

#         for i in range(len(srcs)):
#             srcs[i] = (src_length - len(srcs[i])) * [self.tokenizer.eos_token_id] + srcs[i] + self.tokenizer.encode(['<ans>'])
#         srcs_tensor = torch.LongTensor(srcs)
#         src_length += 1

#         srcs_tensor = srcs_tensor.to(self.device)
#         inputs = srcs_tensor

#         all_output = []
#         for idx in range(self.max_out_len):
#             outputs = self.decoder(inputs)
#             token_logit = outputs[0][:, -1, :]
#             tokens = token_logit.topk(1, dim=1)[1]
#             # mask=tokens==self.tokenizer.pad_token_id
#             all_output.append(tokens)
#             inputs = torch.cat((inputs, tokens), dim=1)
#         all_output = torch.cat(all_output, dim=1)
#         #all_output = self.decode_(all_output)
#         # print (all_output)
#         # print ("all_output:", all_output.size())
#         return all_output, tgts_tensor

#     def decode_(self, outputs):
#         batch_size = outputs.size(0)
#         all_outputs = []
#         for b in range(batch_size):
#             symbols = self.tokenizer.decode(outputs[b])
#             symbols = self.tokenizer.tokenize(symbols)
#             symbols_ = []
#             for token in symbols:
#                 if token == self.tokenizer.eos_token:
#                     break
#                 else:
#                     symbols_.append(token)
#             symbols = symbols_[:]
#             # print ("symbols",symbols)
#             all_outputs.append(symbols)
#         # print (all_outputs)
#         return all_outputs

# class GPT2(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         #self.eval_generate_num = config['eval_generate_num']
#         self.device=config["device"]
#         self.pretrained_model_path = config['pretrained_model_path']

#         self.tokenizer=BertTokenizer.from_pretrained(self.pretrained_model_path)
#         _ =self.tokenizer.add_tokens(config["out_idx2symbol"])
#         self.tokenizer.add_special_tokens({"additional_special_tokens":["<ans>"]})
#         self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

#         self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
#         self.decoder.resize_token_embeddings(len(self.tokenizer))

#         self.padding_token_idx = self.tokenizer.pad_token_id
#         self.max_out_len = config['max_output_len']
#     def forward(self, seq,target=None):

#         if target != None:
#             token_logits,target=self.generate_t(seq,target)
#             return token_logits,target
#         else:
#             all_output=self.generate_without_t(seq)
#             return all_output,None
#     def list2str(self,x):
#         y=''.join(x)
#         return y
#     def generate_t(self,seq,target=None):
#         start_idx=[]
#         output_len=0
#         target_idx=[]
#         batch_size=len(seq)
#         for idx,s in enumerate(seq):
#             seq[idx]=self.tokenizer.tokenize(seq[idx])
#             t=target[idx]
#             target_len=len(t)
#             if output_len<target_len:
#                 output_len=target_len
#             start_idx.append(len(seq[idx])+1)
#             seq[idx]+=(["<ans>"]+t)
#         encoding_dict=self.tokenizer.batch_encode_plus(seq,
#                                             max_length=256,
#                                             pad_to_max_length=True)
#         input_ids=encoding_dict['input_ids']
#         attn_masks=encoding_dict['attention_mask']
#         for b in range(len(start_idx)):
#             target_idx.append(input_ids[b][start_idx[b]:start_idx[b]+output_len])
#         target_idx=torch.tensor(target_idx).to(self.device)
#         input_ids=torch.tensor(input_ids).long().to(self.device)
#         attn_masks=torch.tensor(attn_masks).bool().to(self.device)

#         outputs = self.decoder(input_ids,
#                                 attention_mask=attn_masks)
#         token_logits=[]
#         for b in range(batch_size):
#             token_logits.append(outputs[0][b,start_idx[b]:start_idx[b]+output_len,:])
#         token_logits=torch.stack(token_logits,dim=0)
#         token_logits=token_logits.view(-1,token_logits.size(-1))
#         return token_logits,target_idx
#     def generate_without_t(self,seq):
#         all_output=[]
#         for idx,s in enumerate(seq):
#             seq[idx]=self.tokenizer.tokenize(seq[idx])
#             seq[idx]+=["<ans>"]
#         encoding_dict=self.tokenizer.batch_encode_plus(seq)
#         input_ids=encoding_dict['input_ids']
#         attn_masks=encoding_dict['attention_mask']
#         input_ids=torch.tensor(input_ids).long().to(self.device)
#         attn_masks=torch.tensor(attn_masks).bool().to(self.device)
#         for idx in range(self.max_out_len):
#             outputs = self.decoder(input_ids,
#                                 attention_mask=attn_masks)
#             token_logit=outputs[0][:,-1,:]
#             tokens=token_logit.topk(1,dim=1)[1]
#             mask=tokens==self.tokenizer.pad_token_id
#             all_output.append(tokens)
#             input_ids=torch.cat((input_ids,tokens),dim=1)
#             attn_masks=torch.cat((attn_masks,mask),dim=1)
#         all_output=torch.cat(all_output,dim=1)
#         all_output=self.decode(all_output)
#         return all_output
#     def decode(self,outputs):
#         batch_size=outputs.size(0)
#         all_outputs=[]
#         for b in range(batch_size):
#             symbols=self.tokenizer.decode(outputs[b])
#             symbols=self.tokenizer.tokenize(symbols)
#             all_outputs.append(symbols)
#         return all_outputs

# if __name__ == '__main__':
#     config={
#         "pretrained_model_path":r"C:\Users\74765\.vscode\Programs\MWPToolkit\pretrain\gpt2",
#         "out_sos_token":1,
#         "out_eos_token":2,
#         "max_out_len":20
#     }
#     model=GPT2(config)
#     seq=["5（3）班的同学在母亲节都表达了对妈妈的节日祝福．其中，NUM的同学送了鲜花，NUM的同学给了妈妈一个香香的吻，其余的同学都送上了自制的贺卡．送自制贺卡的同学占全班的几分之几？"]
#     seq=[
#         "5 （ 3 ） 班 的 同学 在 母亲节 都 表达 了 对 妈妈 的 节日 祝福 ． 其中 ， NUM_a 的 同学 送 了 鲜花 ， NUM_b 的 同学 给 了 妈妈 一个 香香的 吻 ， 其余 的 同学 都 送 上 了 自制 的 贺卡 ． 送 自制 贺卡 的 同学 占 全班 的 几分 之 几 ？",
#         #"+ - * / ^",
#     ]
#     target=[["NUM_a","+","NUM_b"],["NUM_a"]]
#     #seq[0]=list(seq[0])
#     #seq[0]=seq[0].split(" ")
#     model.calculate_loss(seq)
#     print(1)
# import torch
# from torch import nn
# from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

# from mwptoolkit.utils.enum_type import SpecialTokens,NumMask
