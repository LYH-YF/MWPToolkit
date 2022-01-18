# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:53
# @File: template_dataset.py


import copy

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset


class TemplateDataset(AbstractDataset):
    """template dataset.

    you need implement:

    TemplateDataset._preprocess()

    TemplateDataset._build_symbol()

    TemplateDataset._build_template_symbol()

    overwrite TemplateDataset._build_vocab() if necessary
    
    """
    def __init__(self, config):
        super().__init__(config)
        self.generate_list = []
        self.operator_list = []
        self.special_token_list = []
        self.copy_nums = 0
        self.out_idx2symbol = []
        self.temp_idx2symbol = []

    def _preprocess(self):
        """
        In this function, you need to implement the codes of data preprocessing.

        Specifically, you need to

        1. format input and output of every data, including trainset, validset and testset.

        2. reset the list variables TemplateDataset.generate_list, TemplateDataset.operator_list and TemplateDataset.special_token_list.

        3. reset the integer variables TemplateDataset.copy_nums

        """
        return super()._preprocess()

    def _build_vocab(self):
        words_count = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count[word] += 1
                except:
                    words_count[word] = 1

        self.in_idx2word = copy.deepcopy(self.special_token_list)

        for key, value in words_count.items():
            if value > self.min_word_keep or "NUM" in key:
                self.in_idx2word.append(key)

        self._build_symbol()
        self._build_template_symbol()
        for symbol in self.out_idx2symbol:
            if symbol in self.in_idx2word:
                continue
            else:
                self.in_idx2word.append(symbol)

        self.in_word2idx = {}
        self.out_symbol2idx = {}
        self.temp_symbol2idx = {}
        for idx, word in enumerate(self.in_idx2word):
            self.in_word2idx[word] = idx
        for idx, symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(self.temp_idx2symbol):
            self.temp_symbol2idx[symbol] = idx

    def _build_symbol(self):
        """
        In this function, you need to implement the codes of building output vocabulary.

        Specifically, you need to

        1. reset the list variables TemplateDataset.out_idx2symbol, append the generating symbols into it.
        """
        raise NotImplementedError

    def _build_template_symbol(self):
        """
        In this function, you need to implement the codes of building output vocabulary for equation template.

        Specifically, you need to

        1. reset the list variables TemplateDataset.temp_idx2symbol, append the generating symbols into it.
        Also, you can do nothing in this function if you don't need template.
        """
        raise NotImplementedError

    def _update_vocab(self, vocab_list):
        index = len(self.in_idx2word)
        for word in vocab_list:
            if word not in self.in_idx2word:
                self.in_idx2word.append(word)
                self.in_word2idx[word] = index
                index += 1

    def get_vocab_size(self):
        return len(self.in_idx2word), len(self.out_idx2symbol)