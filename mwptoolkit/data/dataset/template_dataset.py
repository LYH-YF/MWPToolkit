# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:53
# @File: template_dataset.py


import copy
import json
import os

from mwptoolkit.config.configuration import Config
from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.utils import read_json_data, write_json_data


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

        you should return a dictionary object like
        >>> {
            'generate_list':generate_list,
            'operator_list':operator_list,
            'special_token_list':special_token_list,
            'copy_nums':copy_nums
        }

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

        in_idx2word = copy.deepcopy(self.special_token_list)

        for key, value in words_count.items():
            if value > self.min_word_keep or "NUM" in key:
                in_idx2word.append(key)

        equ_dict = self._build_symbol()
        temp_dict = self._build_template_symbol()
        out_idx2symbol = equ_dict['out_idx2symbol']
        temp_idx2symbol = temp_dict['temp_idx2symbol']
        for symbol in out_idx2symbol:
            if symbol in in_idx2word:
                continue
            else:
                in_idx2word.append(symbol)

        in_word2idx = {}
        out_symbol2idx = {}
        temp_symbol2idx = {}
        for idx, word in enumerate(in_idx2word):
            in_word2idx[word] = idx
        for idx, symbol in enumerate(out_idx2symbol):
            out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(temp_idx2symbol):
            temp_symbol2idx[symbol] = idx

        return_info = {'in_idx2word': in_idx2word, 'in_word2idx': in_word2idx, 'out_idx2symbol': out_idx2symbol,
                       'temp_idx2symbol': temp_idx2symbol, 'out_symbol2idx': out_symbol2idx,
                       'temp_symbol2idx': temp_symbol2idx}
        return_info.update(equ_dict)
        return_info.update(temp_dict)
        return return_info

    def _build_symbol(self):
        """
        In this function, you need to implement the codes of building output vocabulary.

        Specifically, you need to

        1. reset the list variables TemplateDataset.out_idx2symbol, append the generating symbols into it.

        you should return a dictionary object like
        >>> {'out_idx2symbol':out_idx2symbol}
        """
        raise NotImplementedError

    def _build_template_symbol(self):
        """
        In this function, you need to implement the codes of building output vocabulary for equation template.

        Specifically, you need to

        1. reset the list variables TemplateDataset.temp_idx2symbol, append the generating symbols into it.
        Also, you can do nothing in this function if you don't need template.

        ou should return a dictionary object like
        >>> {'temp_idx2symbol':temp_idx2symbol}
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

    def save_dataset(self, save_dir: str):
        """
        save dataset parameters to file.

        :param save_dir: (str) folder which saves the parameter file
        :return:
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        input_vocab_file = os.path.join(save_dir, 'input_vocab.json')
        write_json_data(
            {'in_idx2word': self.in_idx2word},
            input_vocab_file
        )
        output_vocab_file = os.path.join(save_dir, 'output_vocab.json')
        write_json_data(
            {
                'out_idx2symbol': self.out_idx2symbol,
                'temp_idx2symbol': self.temp_idx2symbol
            },
            output_vocab_file
        )
        data_id_file = os.path.join(save_dir, 'data_split.json')
        write_json_data(
            {
                'trainset_id': self.trainset_id,
                'validset_id': self.validset_id,
                'testset_id': self.testset_id,
                'folds_id': self.folds_id
            },
            data_id_file
        )
        json_encoder = json.encoder.JSONEncoder()
        parameters_dict = self.parameters_to_dict()
        not_support_json = []
        not_to_save = ['in_idx2word', 'out_idx2symbol', 'temp_idx2symbol', 'in_word2idx', 'out_symbol2idx',
                       'temp_symbol2idx', 'folds', 'trainset', 'testset', 'validset', 'datas', 'trainset_id',
                       'validset_id', 'testset_id', 'folds_id']
        for key, value in parameters_dict.items():
            try:
                json_encoder.encode({key: value})
            except TypeError:
                not_support_json.append(key)
        for key in not_support_json:
            del parameters_dict[key]
        for key in not_to_save:
            del parameters_dict[key]
        parameter_file = os.path.join(save_dir, 'dataset.json')
        write_json_data(parameters_dict, parameter_file)

    @classmethod
    def load_from_pretrained(cls, pretrained_dir: str, resume_training=False):
        """
        load dataset parameters from file.

        :param pretrained_dir: (str) folder which saved the parameter file
        :param resume_training: (bool) load parameter for resuming training or not.
        :return: an instantiated object
        """
        config = Config.load_from_pretrained(pretrained_dir)
        dataset = cls(config)

        input_vocab_file = os.path.join(pretrained_dir, 'input_vocab.json')
        output_vocab_file = os.path.join(pretrained_dir, 'output_vocab.json')
        parameter_file = os.path.join(pretrained_dir, 'dataset.json')
        data_id_file = os.path.join(pretrained_dir, 'data_split.json')

        input_vocab = read_json_data(input_vocab_file)
        output_vocab = read_json_data(output_vocab_file)
        parameter_dict = read_json_data(parameter_file)
        data_id_dict = read_json_data(data_id_file)

        in_idx2word = input_vocab['in_idx2word']
        out_idx2symbol = output_vocab['out_idx2symbol']
        temp_idx2symbol = output_vocab['temp_idx2symbol']

        in_word2idx = {}
        out_symbol2idx = {}
        temp_symbol2idx = {}
        for idx, word in enumerate(in_idx2word):
            in_word2idx[word] = idx
        for idx, symbol in enumerate(out_idx2symbol):
            out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(temp_idx2symbol):
            temp_symbol2idx[symbol] = idx

        setattr(dataset, 'in_idx2word', in_idx2word)
        setattr(dataset, 'out_idx2symbol', out_idx2symbol)
        setattr(dataset, 'temp_idx2symbol', temp_idx2symbol)
        setattr(dataset, 'in_word2idx', in_word2idx)
        setattr(dataset, 'out_symbol2idx', out_symbol2idx)
        setattr(dataset, 'temp_symbol2idx', temp_symbol2idx)
        for key, value in parameter_dict.items():
            setattr(dataset, key, value)
        for key, value in data_id_dict.items():
            setattr(dataset, key, value)
        if resume_training:
            if config['k_fold']:
                setattr(dataset, 'fold_t', config['fold_t'])
                setattr(dataset, 'the_fold_t', config['fold_t'] - 1)
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
            else:
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
        else:
            setattr(dataset, 'from_pretrained', True)
            setattr(dataset, 'pretrained_dir', pretrained_dir)
        dataset.reset_dataset()
        return dataset

    def __load_pretrained_parameters(self):
        if self.k_fold:
            load_dir = os.path.join(self.pretrained_dir, 'fold{}'.format(self.fold_t))
        else:
            load_dir = self.pretrained_dir

        input_vocab_file = os.path.join(load_dir, 'input_vocab.json')
        output_vocab_file = os.path.join(load_dir, 'output_vocab.json')
        parameter_file = os.path.join(load_dir, 'dataset.json')

        input_vocab = read_json_data(input_vocab_file)
        output_vocab = read_json_data(output_vocab_file)
        parameter_dict = read_json_data(parameter_file)

        in_idx2word = input_vocab['in_idx2word']
        out_idx2symbol = output_vocab['out_idx2symbol']
        temp_idx2symbol = output_vocab['temp_idx2symbol']

        in_word2idx = {}
        out_symbol2idx = {}
        temp_symbol2idx = {}
        for idx, word in enumerate(in_idx2word):
            in_word2idx[word] = idx
        for idx, symbol in enumerate(out_idx2symbol):
            out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(temp_idx2symbol):
            temp_symbol2idx[symbol] = idx

        setattr(self, 'in_idx2word', in_idx2word)
        setattr(self, 'out_idx2symbol', out_idx2symbol)
        setattr(self, 'temp_idx2symbol', temp_idx2symbol)
        setattr(self, 'in_word2idx', in_word2idx)
        setattr(self, 'out_symbol2idx', out_symbol2idx)
        setattr(self, 'temp_symbol2idx', temp_symbol2idx)
        for key, value in parameter_dict.items():
            setattr(self, key, value)
