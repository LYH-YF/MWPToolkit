# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2022/3/25 13:46
# @File: dataloader_gpt2.py
# @Update Time: 2022/3/25 13:46
import math

import torch
from typing import List

from mwptoolkit.config import Config
from mwptoolkit.data.dataset import PretrainDataset
from mwptoolkit.data.dataloader.template_dataloader import TemplateDataLoader
from mwptoolkit.utils.enum_type import FixType, SpecialTokens


def get_num_mask(num_size_batch, generate_nums):
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    return num_mask


class DataLoaderGPT2(TemplateDataLoader):
    """dataloader class for pre-train model.
    """

    def __init__(self, config: Config, dataset: PretrainDataset):
        """

        :param config:
        :param dataset:

        expected that config includes these parameters below:

        model (str): model name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.

        train_batch_size (int): the training batch size.

        test_batch_size (int): the testing batch size.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        max_len (int|None): max input length.

        max_equ_len (int|None): max output length.

        add_sos (bool): add sos token at the head of input sequence.

        add_eos (bool): add eos token at the tail of input sequence.

        device (torch.device):
        """
        super().__init__(config, dataset)
        # self.dataset=PretrainDataset(config)
        # dataset=PretrainDataset(config)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.in_pad_token = dataset.tokenizer.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)
        self.in_unk_token = dataset.tokenizer.convert_tokens_to_ids(SpecialTokens.UNK_TOKEN)
        self.in_eos_token = dataset.tokenizer.convert_tokens_to_ids(SpecialTokens.EOS_TOKEN)

        if self.symbol_for_tree or self.equation_fix == FixType.MultiWayTree:
            self.out_pad_token = self.in_pad_token
            self.out_unk_token = dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
            self.out_eos_token = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
            self.temp_pad_token = self.in_pad_token
            self.temp_unk_token = dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
            self.temp_eos_token = dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        else:
            if self.share_vocab:
                self.out_pad_token = self.in_pad_token
                self.out_unk_token = self.in_unk_token
                self.out_eos_token = self.in_eos_token
                self.temp_pad_token = self.in_pad_token
                self.temp_unk_token = self.in_unk_token
                self.temp_eos_token = self.in_eos_token
            else:
                self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.out_unk_token = dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
                self.out_eos_token = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
                self.temp_pad_token = dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.temp_unk_token = dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
                self.temp_eos_token = dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        self.__init_batches()

    def _word2idx(self, sentence):
        sentence_idx = []

        sentence_idx = self.dataset.tokenizer.convert_tokens_to_ids(sentence)
        # sentence_idx = self.dataset.tokenizer.encode(sentence,add_special_token=False)

        return sentence_idx

    def _equ_symbol2idx(self, equation):
        equ_idx = []
        if self.equation_fix == FixType.MultiWayTree:
            for symbol in equation:
                if isinstance(symbol, list):
                    sub_equ_idx = self._equ_symbol2idx(symbol)
                    equ_idx.append(sub_equ_idx)
                else:
                    if self.share_vocab:
                        idx = self.dataset.tokenizer.convert_tokens_to_ids(symbol)
                    else:
                        try:
                            idx = self.dataset.out_symbol2idx[symbol]
                        except:
                            idx = self.out_unk_token
                    equ_idx.append(idx)
        else:
            for symbol in equation:
                if self.share_vocab:
                    idx = self.dataset.tokenizer.convert_tokens_to_ids(symbol)

                else:
                    try:
                        idx = self.dataset.out_symbol2idx[symbol]
                    except:
                        idx = self.out_unk_token
                equ_idx.append(idx)
        return equ_idx

    def __build_batch(self, batch_data):
        ques_batch = []
        equ_batch = []
        temp_batch = []
        ques_source_batch = []
        equ_source_batch = []
        temp_source_batch = []
        ques_source_1_batch = []
        infix_equ_batch = []

        num_list_batch = []
        num_pos_batch = []

        id_batch = []
        ans_batch = []

        equ_len_batch = []
        ques_len_batch = []

        num_stack_batch = []

        group_nums_batch = []

        batch_data = sorted(batch_data, key=lambda x: len(x['question']), reverse=True)
        for data in batch_data:
            sentence = data["question"]
            equation = data["equation"]

            # question word to index
            if self.add_sos:
                sentence = [SpecialTokens.SOS_TOKEN] + sentence
            if self.add_eos:
                sentence = sentence + [SpecialTokens.EOS_TOKEN]
            sentence = sentence + ['<ans>']
            ques_tensor = self._word2idx(sentence)

            # equation symbol to index
            equation.append(SpecialTokens.EOS_TOKEN)
            equ_tensor = self._equ_symbol2idx(equation)

            equ_len_batch.append(len(equ_tensor))
            ques_len_batch.append(len(ques_tensor))
            ques_batch.append(ques_tensor)
            equ_batch.append(equ_tensor)

            # question / equation to string
            ques_source = ' '.join(sentence)
            if self.equation_fix == FixType.MultiWayTree:
                equ_source = ' '
            else:
                equ_source = ' '.join(equation)
            ques_source_batch.append(ques_source)
            equ_source_batch.append(equ_source)
            ques_source_1_batch.append(data["ques source 1"])
            # infix equation
            infix_equ_batch.append(data["infix equation"])
            # quantity list
            num_list_batch.append(data["number list"])
            # quantity position
            if self.add_sos:
                num_pos = [pos + 1 for pos in
                           data["number position"]]  # pos plus one because of adding <SOS> at the head of sentence
            else:
                num_pos = [pos for pos in data["number position"]]
            num_pos_batch.append(num_pos)
            # question id and answer
            id_batch.append(data["id"])
            ans_batch.append(data["ans"])
            try:
                group_nums_batch.append(data["group nums"])
            except:
                group_nums_batch.append([])

            num_stack_batch.append(self._build_num_stack(equation, data["number list"]))

        # padding batch question
        ques_batch = self._pad_input_batch(ques_batch, ques_len_batch)
        if self.max_len is not None:
            ques_len_batch = [self.max_len if l > self.max_len else l for l in ques_len_batch]
        # padding batch equation
        if self.equation_fix == FixType.MultiWayTree:
            pass
        else:
            equ_batch = self._pad_output_batch(equ_batch, equ_len_batch)
        # question mask
        ques_mask_batch = self._get_input_mask(ques_len_batch)
        # equation mask
        equ_mask_batch = self._get_mask(equ_len_batch)
        # quantity count
        num_size_batch = [len(num_pos) for num_pos in num_pos_batch]
        # quantity mask
        num_mask_batch = get_num_mask(num_size_batch, self.dataset.generate_list)

        new_group_nums_batch = []
        for group_nums in group_nums_batch:
            new_group_nums = []
            for group_num in group_nums:
                new_group_num = []
                for pos in group_num:
                    if self.add_sos:
                        new_group_num.append(pos + 1)
                    else:
                        new_group_num.append(pos)
                new_group_nums.append(new_group_num)
            new_group_nums_batch.append(new_group_nums)

        return {
            "question": ques_batch,
            "equation": equ_batch,
            "template": temp_batch,
            "ques len": ques_len_batch,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "id": id_batch,
            "num mask": num_mask_batch,
            "ques mask": ques_mask_batch,
            "equ mask": equ_mask_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "num size": num_size_batch,
            "ques_source": ques_source_batch,
            "equ_source": equ_source_batch,
            "temp_source": temp_source_batch,
            "ques source 1": ques_source_1_batch,
            "group nums": new_group_nums_batch,
            "infix equation": infix_equ_batch,
        }

    def __init_batches(self):
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        for set_type in ['train', 'valid', 'test']:
            if set_type == 'train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
            elif set_type == 'valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
            elif set_type == 'test':
                datas = self.dataset.testset
                batch_size = self.test_batch_size
            else:
                raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
            num_total = len(datas)
            batch_num = math.ceil(num_total / batch_size)
            for batch_i in range(batch_num):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                if end_idx <= num_total:
                    batch_data = datas[start_idx:end_idx]
                else:
                    batch_data = datas[start_idx:num_total]
                built_batch = self.__build_batch(batch_data)
                if set_type == 'train':
                    self.trainset_batches.append(built_batch)
                elif set_type == 'valid':
                    self.validset_batches.append(built_batch)
                elif set_type == 'test':
                    self.testset_batches.append(built_batch)
                else:
                    raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)

    def build_batch_for_predict(self, batch_data: List[dict]):
        for idx, data in enumerate(batch_data):
            data['equation'] = []
            data['template'] = []
            data['infix equation'] = []
            data['ans'] = None
            if data.get('id', None) is None:
                data['id'] = 'temp_{}'.format(idx)
        batch = self.__build_batch(batch_data)
        del batch['equation']
        del batch['template']
        del batch['equ len']
        del batch['equ mask']
        del batch['ans']
        del batch['equ_source']
        del batch['temp_source']
        del batch['infix equation']

        return batch

    def _pad_input_batch(self, batch_seq, batch_seq_len):
        if self.max_len is not None:
            max_length = self.max_len
        else:
            max_length = max(batch_seq_len)
        for idx, length in enumerate(batch_seq_len):
            if max_length >= length:
                batch_seq[idx] = (max_length - length - 1) * [self.in_eos_token] + batch_seq[idx]
            else:
                if self.add_sos and self.add_eos:
                    seq_i = batch_seq[idx]
                    batch_seq[idx] = [seq_i[0]] + seq_i[1:max_length-2] + [seq_i[-2]] + [seq_i[-1]]  # [<s>,tokens,<e>,<ans>]
                else:
                    batch_seq[idx] = batch_seq[idx][:max_length -1] + [batch_seq[idx][-1]]   # [tokens,<ans>]
        return batch_seq

    def _pad_output_batch(self, batch_target, batch_target_len):
        if self.max_equ_len is not None:
            max_length = self.max_equ_len
        else:
            max_length = max(batch_target_len)
        for idx, length in enumerate(batch_target_len):
            if length < max_length:
                batch_target[idx] += (max_length - length) * [self.out_eos_token]
            else:
                batch_target[idx] = batch_target[idx][:max_length]
        return batch_target
