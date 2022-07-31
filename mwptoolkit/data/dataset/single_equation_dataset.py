# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:45
# @File: single_equation_dataset.py


import os
import copy
import json
import warnings
from logging import getLogger
import re
from collections import Counter

import torch

from mwptoolkit.config.configuration import Config
from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import preprocess_ept_dataset_
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_multi_way_tree, postfix_parser
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tool.sentence_operator import deprel_tree_to_file, get_group_nums_, span_level_deprel_tree_to_file, get_span_level_deprel_tree_, get_deprel_tree_
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.enum_type import MaskSymbol, NumMask, SpecialTokens, FixType, Operators, DatasetName, EPT
from mwptoolkit.utils.enum_type import OPERATORS, SPECIAL_TOKENS
from transformers import AutoTokenizer

from mwptoolkit.utils.utils import write_json_data, read_json_data


class SingleEquationDataset(AbstractDataset):
    """single-equation dataset

    preprocess dataset when running single-equation task.
    """
    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

        rule1 (bool): convert equation according to rule 1.

        rule2 (bool): convert equation according to rule 2.

        parse_tree_file_name (str|None): the name of the file to save parse tree information.

        model (str): model name.

        dataset (str): dataset name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.
        
        dataset_dir or dataset_path (str): the road path of dataset folder.

        language (str): a property of dataset, the language of dataset.

        single (bool): a property of dataset, the equation of dataset is single or not.

        linear (bool): a property of dataset, the equation of dataset is linear or not.

        source_equation_fix (str): [infix | postfix | prefix], a property of dataset, the source format of equation of dataset.

        rebuild (bool): when loading additional dataset information, this can decide to build information anew or load information built before.

        validset_divide (bool): whether to split validset. if True, the dataset is split to trainset-validset-testset. if False, the dataset is split to trainset-testset.

        mask_symbol (str): [NUM | number], the symbol to mask numbers in equation.

        min_word_keep (int): in dataset, words that count greater than the value, will be kept in input vocabulary.

        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        k_fold (int|None): if it's an integer, it indicates to run k-fold cross validation. if it's None, it indicates to run trainset-validset-testset split.

        read_local_folds (bool): when running k-fold cross validation, if True, then loading split folds from dataset folder. if False, randomly split folds.

        shuffle (bool): whether to shuffle trainset before training.

        device (torch.device):

        resume_training or resume (bool):
        """
        super().__init__(config)
        self.module_name = 'SingleEquationDataset'

        self.rule1 = config["rule1"]
        self.rule2 = config["rule2"]
        self.parse_tree_path = config['parse_tree_file_name']
        if self.parse_tree_path is not None:
            self.parse_tree_path = os.path.join(self.dataset_path,self.parse_tree_path + '.json')
            if not os.path.isabs(self.parse_tree_path):
                self.parse_tree_path = os.path.join(os.getcwd(),self.parse_tree_path)

        self.in_idx2word = []
        self.out_idx2symbol = []
        self.temp_idx2symbol = []
        self.in_word2idx = {}
        self.out_symbol2idx = {}
        self.temp_symbol2idx = {}

        self.copy_nums = 0
        self.generate_list = []
        self.operator_list = []
        self.operator_nums = len(self.operator_list)
    
    def _preprocess(self):
        transfer = number_transfer
        
        self.trainset, generate_list, train_copy_nums,unk_symbols = transfer(self.trainset, self.dataset, 'single_equation', self.mask_symbol, self.min_generate_keep,self.linear)
        self.validset, _g, valid_copy_nums,_ = transfer(self.validset, self.dataset, 'single_equation', self.mask_symbol, self.min_generate_keep,self.linear)
        self.testset, _g, test_copy_nums,_ = transfer(self.testset, self.dataset, 'single_equation', self.mask_symbol, self.min_generate_keep,self.linear)
        
        target_equation_fix=self.equation_fix if self.equation_fix else FixType.Infix
        source_equation_fix=self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if self.rule1:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-infix-equation datasets may not support EN rule1 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule1_process(k=max([train_copy_nums, valid_copy_nums, test_copy_nums]))
            else:
                warnings.warn("non-linear or non-single datasets may not support EN rule1 process, already ignored it. ")
                #raise Warning("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")

        if self.rule2:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-infix-equation datasets may not support EN rule2 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule2_process()
            else:
                warnings.warn("non-linear or non-single datasets may not support EN rule2 process, already ignored it. ")
                #raise Warning("non-linear or non-single datasets may not surport en rule2 process, already ignored it. ")
            
        if source_equation_fix == target_equation_fix:
            fix = None
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.Prefix:
            fix = from_infix_to_prefix
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.Postfix:
            fix = from_infix_to_postfix
        elif source_equation_fix == FixType.Prefix and target_equation_fix == FixType.Postfix:
            fix = from_prefix_to_postfix
        elif source_equation_fix == FixType.Prefix and target_equation_fix == FixType.Infix:
            fix = from_prefix_to_infix
        elif source_equation_fix == FixType.Postfix and target_equation_fix == FixType.Infix:
            fix = from_postfix_to_infix
        elif source_equation_fix == FixType.Postfix and target_equation_fix == FixType.Prefix:
            fix = from_postfix_to_prefix
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.MultiWayTree:
            fix = from_infix_to_multi_way_tree
        else:
            raise NotImplementedError("the type of equation fix ({}) is not implemented.".format(self.equation_fix))

        self.fix_process(fix)
        self.operator_mask_process()

        generate_list = unk_symbols + generate_list
        if self.symbol_for_tree:
            copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        elif self.model.lower() in ['saligned']:
            copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            copy_nums = train_copy_nums
        operator_list = copy.deepcopy(Operators.Single)
        if self.dataset in [DatasetName.mawps]:
            operator_list.append('=')
        operator_nums = len(operator_list)

        # graph preprocess
        use_gpu = True if self.device == torch.device('cuda') else False
        if self.model.lower() in ['graph2treeibm']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset, token_list =\
                    get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
                deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                        self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset, token_list =\
                    get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)

        if self.model.lower() in ['graph2tree']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset =\
                    get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
                deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                        self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset =\
                    get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)

        return {'generate_list': generate_list, 'copy_nums': copy_nums, 'operator_list': operator_list,
                'operator_nums': operator_nums}

    def _build_vocab(self):
        words_count = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count[word] += 1
                except:
                    words_count[word] = 1
        in_idx2word = [SpecialTokens.PAD_TOKEN,SpecialTokens.SOS_TOKEN,SpecialTokens.EOS_TOKEN,SpecialTokens.UNK_TOKEN]

        for key, value in words_count.items():
            if value > self.min_word_keep or "NUM" in key:
                in_idx2word.append(key)
        
        if self.symbol_for_tree:
            equ_dict = self._build_symbol_for_tree()
            temp_dict = self._build_template_symbol_for_tree()
        elif self.equation_fix == FixType.MultiWayTree:
            equ_dict = self._build_symbol_for_multi_way_tree()
            temp_dict = self._build_template_symbol_for_multi_way_tree()
        else:
            equ_dict = self._build_symbol()
            temp_dict = self._build_template_symbol()
        out_idx2symbol = equ_dict['out_idx2symbol']
        temp_idx2symbol = temp_dict['temp_idx2symbol']
        num_start = equ_dict['num_start']
        temp_num_start = temp_dict['temp_num_start']

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

        return {'in_idx2word': in_idx2word, 'in_word2idx': in_word2idx, 'out_idx2symbol': out_idx2symbol,
                'temp_idx2symbol': temp_idx2symbol, 'out_symbol2idx': out_symbol2idx,
                'temp_symbol2idx': temp_symbol2idx, 'num_start': num_start,
                'temp_num_start': temp_num_start}

    def _build_symbol_for_tree(self):
        out_idx2symbol = copy.deepcopy(self.operator_list)
        num_start = len(out_idx2symbol)
        out_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        out_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'out_idx2symbol': out_idx2symbol, 'num_start': num_start}

    def _build_symbol_for_multi_way_tree(self):
        out_idx2symbol = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.NON_TOKEN]
        out_idx2symbol += Operators.Single
        num_start = len(out_idx2symbol)
        out_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))
        out_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'out_idx2symbol': out_idx2symbol, 'num_start': num_start}

    def _build_symbol(self):
        if self.share_vocab:
            out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        else:
            out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        if self.model.lower() in ['hms']:
            out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        num_start = len(out_idx2symbol)
        out_idx2symbol += self.generate_list
        if self.model.lower() in ['hms']:
            out_idx2symbol += [SpecialTokens.UNK_TOKEN]
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))
        for data in self.trainset:
            words_list = data["equation"]
            for word in words_list:
                if word in out_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    out_idx2symbol.insert(num_start, word)
                    num_start += 1
                    continue
                else:
                    out_idx2symbol.append(word)
        if self.model.lower() in ['hms']:
            return {'out_idx2symbol': out_idx2symbol, 'num_start': num_start}
        out_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'out_idx2symbol': out_idx2symbol, 'num_start': num_start}

    def _build_template_symbol(self):
        if self.share_vocab:
            temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + [SpecialTokens.OPT_TOKEN]
        else:
            temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + [SpecialTokens.OPT_TOKEN]

        temp_num_start = len(temp_idx2symbol)
        temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        for data in self.trainset:
            words_list = data["template"]
            for word in words_list:
                if word in temp_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    temp_idx2symbol.insert(temp_num_start, word)
                    temp_num_start += 1
                    continue
                else:
                    temp_idx2symbol.append(word)
        temp_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'temp_idx2symbol':temp_idx2symbol,'temp_num_start':temp_num_start}

    def _build_template_symbol_for_multi_way_tree(self):
        temp_idx2symbol = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.NON_TOKEN]
        temp_idx2symbol += [SpecialTokens.OPT_TOKEN]
        temp_num_start = len(temp_idx2symbol)
        temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        temp_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'temp_idx2symbol': temp_idx2symbol, 'temp_num_start': temp_num_start}

    def _build_template_symbol_for_tree(self):
        temp_idx2symbol = [SpecialTokens.OPT_TOKEN]
        temp_num_start = len(temp_idx2symbol)
        temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        temp_idx2symbol += [SpecialTokens.UNK_TOKEN]
        return {'temp_idx2symbol': temp_idx2symbol, 'temp_num_start': temp_num_start}

    def _update_vocab(self, vocab_list):
        index = len(self.in_idx2word)
        for word in vocab_list:
            if word not in self.in_idx2word:
                self.in_idx2word.append(word)
                self.in_word2idx[word] = index
                index += 1

    def get_vocab_size(self):
        """
        Returns:
            (tuple(int, int)): the length of input vocabulary and output symbols
        """
        return len(self.in_idx2word), len(self.out_idx2symbol)

    def save_dataset(self,save_dir:str):
        """
        save dataset parameters to file.

        :param save_dir: (str) folder which saves the parameter file
        :return:
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        input_vocab_file = os.path.join(save_dir,'input_vocab.json')
        write_json_data(
            {'in_idx2word': self.in_idx2word},
            input_vocab_file
        )
        output_vocab_file = os.path.join(save_dir,'output_vocab.json')
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
                'trainset_id':self.trainset_id,
                'validset_id':self.validset_id,
                'testset_id':self.testset_id,
                'folds_id':self.folds_id
            },
            data_id_file
        )
        json_encoder = json.encoder.JSONEncoder()
        parameters_dict = self.parameters_to_dict()
        not_support_json=[]
        not_to_save = ['in_idx2word', 'out_idx2symbol', 'temp_idx2symbol', 'in_word2idx', 'out_symbol2idx',
                       'temp_symbol2idx', 'folds', 'trainset', 'testset', 'validset', 'datas', 'trainset_id','validset_id','testset_id','folds_id']
        for key,value in parameters_dict.items():
            try:
                json_encoder.encode({key:value})
            except TypeError:
                not_support_json.append(key)
        for key in not_support_json:
            del parameters_dict[key]
        for key in not_to_save:
            del parameters_dict[key]
        parameter_file = os.path.join(save_dir, 'dataset.json')
        write_json_data(parameters_dict,parameter_file)

    @classmethod
    def load_from_pretrained(cls,pretrained_dir:str,resume_training=False):
        """
        load dataset parameters from file.

        :param pretrained_dir: (str) folder which saved the parameter file
        :param resume_training: (bool) load parameter for resuming training or not.
        :return: an instantiated object
        """
        config = Config.load_from_pretrained(pretrained_dir)
        dataset = SingleEquationDataset(config)

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
        for key,value in data_id_dict.items():
            setattr(dataset, key, value)
        if resume_training:
            if config['k_fold']:
                setattr(dataset,'fold_t',config['fold_t'])
                setattr(dataset,'the_fold_t',config['fold_t']-1)
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
            else:
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
        else:
            setattr(dataset,'from_pretrained', True)
            setattr(dataset,'pretrained_dir', pretrained_dir)
        dataset.reset_dataset()
        return dataset

