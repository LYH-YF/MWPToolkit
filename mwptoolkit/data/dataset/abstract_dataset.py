# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:32:49
# @File: abstract_dataset.py


import random
import os
import copy
import re
import torch

from mwptoolkit.config.configuration import Config
from mwptoolkit.utils.utils import read_json_data, write_json_data
from mwptoolkit.utils.preprocess_tools import get_group_nums, get_deprel_tree, get_span_level_deprel_tree
from mwptoolkit.utils.preprocess_tools import id_reedit
from mwptoolkit.utils.preprocess_tool.equation_operator import from_postfix_to_infix, from_prefix_to_infix, operator_mask,EN_rule1_stat,EN_rule2
from mwptoolkit.utils.enum_type import DatasetName,FixType


class AbstractDataset(object):
    """abstract dataset

    the base class of dataset class
    """
    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

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
        super().__init__()
        self.model = config["model"]
        self.dataset = config["dataset"]
        self.equation_fix = config["equation_fix"]
        
        self.dataset_path = config['dataset_dir'] if config['dataset_dir'] else config["dataset_path"]
        self.language = config["language"]
        self.single = config["single"]
        self.linear = config["linear"]
        self.source_equation_fix = config["source_equation_fix"]

        self.rebuild = config['rebuild']
        self.validset_divide = config["validset_divide"]
        
        self.mask_symbol = config["mask_symbol"]
        self.min_word_keep = config["min_word_keep"]
        self.min_generate_keep = config["min_generate_keep"]
        self.symbol_for_tree = config["symbol_for_tree"]
        self.share_vocab = config["share_vocab"]
        
        self.k_fold = config["k_fold"]
        self.read_local_folds = config["read_local_folds"]
        self.shuffle = config["shuffle"]
        
        self.device = config["device"]

        self.resume_training = config['resume_training'] if config['resume_training'] else config['resume']

        self.max_span_size = 1
        self.fold_t = 0
        self.the_fold_t = -1
        self.from_pretrained = False
        self.datas = []

        self.trainset = []
        self.validset = []
        self.testset = []
        self.validset_id = []
        self.trainset_id = []
        self.testset_id = []
        self.folds = []
        self.folds_id = []
        if self.k_fold:
            self._load_k_fold_dataset()
        else:
            self._load_dataset()

    def _load_all_data(self):
        trainset_file = os.path.join(self.dataset_path, 'trainset.json')
        validset_file = os.path.join(self.dataset_path, 'validset.json')
        testset_file = os.path.join(self.dataset_path, 'testset.json')

        if os.path.isabs(trainset_file):
            trainset = read_json_data(trainset_file)
        else:
            trainset = read_json_data(os.path.join(os.getcwd(), trainset_file))
        if os.path.isabs(validset_file):
            validset = read_json_data(validset_file)
        else:
            validset = read_json_data(os.path.join(os.getcwd(), validset_file))
        if os.path.isabs(testset_file):
            testset = read_json_data(testset_file)
        else:
            testset = read_json_data(os.path.join(os.getcwd(), testset_file))

        return trainset + validset + testset

    def _load_dataset(self):
        '''
        read dataset from files
        '''
        if self.trainset_id and self.testset_id:
            self._init_split_from_id()
        else:
            trainset_file = os.path.join(self.dataset_path,'trainset.json')
            validset_file = os.path.join(self.dataset_path, 'validset.json')
            testset_file = os.path.join(self.dataset_path, 'testset.json')
        
            if os.path.isabs(trainset_file):
                self.trainset = read_json_data(trainset_file)
            else:
                self.trainset = read_json_data(os.path.join(os.getcwd(),trainset_file))
            if os.path.isabs(validset_file):
                self.validset = read_json_data(validset_file)
            else:
                self.validset = read_json_data(os.path.join(os.getcwd(),validset_file))
            if os.path.isabs(testset_file):
                self.testset = read_json_data(testset_file)
            else:
                self.testset = read_json_data(os.path.join(os.getcwd(),testset_file))

            if self.validset_divide is not True:
                self.testset = self.validset + self.testset
                self.validset = []

            if self.dataset in [DatasetName.hmwp]:
                self.trainset,self.validset,self.testset = id_reedit(self.trainset, self.validset, self.testset)

            self._init_id_from_split()

    def _load_fold_dataset(self):
        """read one fold of dataset from file. 
        """
        trainset_file = os.path.join(self.dataset_path, "trainset_fold{}.json".format(self.fold_t))
        testset_file = os.path.join(self.dataset_path, "testset_fold{}.json".format(self.fold_t))

        if os.path.isabs(trainset_file):
            self.trainset = read_json_data(trainset_file)
        else:
            self.trainset = read_json_data(os.path.join(os.getcwd(),trainset_file))
        if os.path.isabs(trainset_file):
            self.testset = read_json_data(testset_file)
        else:
            self.testset = read_json_data(os.path.join(os.getcwd(),testset_file))
        self.validset = []

    def _load_k_fold_dataset(self):
        if self.folds_id:
            self._init_folds_form_id()
        else:
            if self.read_local_folds is not True:
                datas = self._load_all_data()
                random.shuffle(datas)
                step_size = int(len(datas) / self.k_fold)
                folds = []
                for split_fold in range(self.k_fold - 1):
                    fold_start = step_size * split_fold
                    fold_end = step_size * (split_fold + 1)
                    folds.append(datas[fold_start:fold_end])
                folds.append(datas[(step_size * (self.k_fold - 1)):])
            else:
                folds = []
                for fold_t in range(self.k_fold):
                    testset_file = self.dataset_path + "/testset_fold{}.json".format(fold_t)
                    if os.path.isabs(testset_file):
                        folds.append(read_json_data(testset_file))
                    else:
                        folds.append(read_json_data(os.path.join(os.getcwd(), testset_file)))
            self.folds = folds
            self._init_id_from_folds()

    def _init_split_from_id(self):
        if self.dataset == DatasetName.asdiv_a:
            id_key = '@ID'
        elif self.dataset == DatasetName.mawps_single:
            id_key = 'iIndex'
        elif self.dataset == DatasetName.SVAMP:
            id_key = 'ID'
        else:
            id_key = 'id'
        datas = self._load_all_data()
        self.trainset = []
        self.validset = []
        self.testset = []
        for data_id in self.trainset_id:
            for idx, data in enumerate(datas):
                if data_id == data[id_key]:
                    self.trainset.append(data)
                    datas.pop(idx)
                    break
        for data_id in self.validset_id:
            for idx, data in enumerate(datas):
                if data_id == data[id_key]:
                    self.validset.append(data)
                    datas.pop(idx)
                    break
        for data_id in self.testset_id:
            for idx, data in enumerate(datas):
                if data_id == data[id_key]:
                    self.testset.append(data)
                    datas.pop(idx)
                    break

    def _init_id_from_split(self):
        if self.dataset == DatasetName.asdiv_a:
            id_key = '@ID'
        elif self.dataset == DatasetName.mawps_single:
            id_key = 'iIndex'
        elif self.dataset == DatasetName.SVAMP:
            id_key = 'ID'
        else:
            id_key = 'id'
        self.trainset_id = []
        self.validset_id = []
        self.testset_id = []
        for data in self.trainset:
            self.trainset_id.append(data[id_key])
        for data in self.validset:
            self.validset_id.append(data[id_key])
        for data in self.testset:
            self.testset_id.append(data[id_key])

    def _init_folds_form_id(self):
        if self.dataset == DatasetName.asdiv_a:
            id_key = '@ID'
        elif self.dataset == DatasetName.mawps_single:
            id_key = 'iIndex'
        elif self.dataset == DatasetName.SVAMP:
            id_key = 'ID'
        else:
            id_key = 'id'
        datas = self._load_all_data()
        self.folds = []
        for fold_t_id in self.folds_id:
            split_fold_data = []
            for data_id in fold_t_id:
                for idx, data in enumerate(datas):
                    if data_id == data[id_key]:
                        split_fold_data.append(data)
                        datas.pop(idx)
                        break
            self.folds.append(split_fold_data)

    def _init_id_from_folds(self):
        if self.dataset == DatasetName.asdiv_a:
            id_key = '@ID'
        elif self.dataset == DatasetName.mawps_single:
            id_key = 'iIndex'
        elif self.dataset == DatasetName.SVAMP:
            id_key = 'ID'
        else:
            id_key = 'id'
        self.folds_id = []
        for split_fold_data in self.folds:
            fold_id = []
            for data in split_fold_data:
                fold_id.append(data[id_key])
            self.folds_id.append(fold_id)

    def reset_dataset(self):
        if self.k_fold:
            self._load_k_fold_dataset()
        else:
            self._load_dataset()

    def fix_process(self, fix):
        r"""equation infix/postfix/prefix process.

        Args:
            fix (function): a function to make infix, postfix, prefix or None  
        """
        source_equation_fix=self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if fix != None:
            for idx, data in enumerate(self.trainset):
                if source_equation_fix==FixType.Prefix:
                    self.trainset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.trainset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.trainset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.validset):
                if source_equation_fix==FixType.Prefix:
                    self.validset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.validset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.validset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.testset):
                if source_equation_fix==FixType.Prefix:
                    self.testset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.testset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.testset[idx]["equation"] = fix(data["equation"])
        else:
            for idx, data in enumerate(self.trainset):
                if source_equation_fix==FixType.Prefix:
                    self.trainset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.trainset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.validset):
                if source_equation_fix==FixType.Prefix:
                    self.validset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.validset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.testset):
                if source_equation_fix==FixType.Prefix:
                    self.testset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.testset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])

    def operator_mask_process(self):
        """operator mask process of equation.
        """
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["template"] = operator_mask(data["equation"])

    def en_rule1_process(self, k):
        rule1_list = EN_rule1_stat(self.trainset, k)
        for idx, data in enumerate(self.trainset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.trainset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.validset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.validset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.testset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.testset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break

    def en_rule2_process(self):
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["equation"] = EN_rule2(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["equation"] = EN_rule2(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["equation"] = EN_rule2(data["equation"])

    def cross_validation_load(self, k_fold, start_fold_t=None):
        r"""dataset load for cross validation

        Build folds for cross validation. Choose one of folds for testset and other folds for trainset.
        
        Args:
            k_fold (int): the number of folds, also the cross validation parameter k.
            start_fold_t (int): default None, training start from the training of t-th time.
        
        Returns:
            Generator including current training index of cross validation.
        """
        if k_fold <= 1:
            raise ValueError("the cross validation parameter k shouldn't be less than one, it should be greater than one")
        if self.read_local_folds is not True:
            self._load_dataset()
            self.datas = self.trainset + self.validset + self.testset
            random.shuffle(self.datas)
            step_size = int(len(self.datas) / k_fold)
            folds = []
            for split_fold in range(k_fold - 1):
                fold_start = step_size * split_fold
                fold_end = step_size * (split_fold + 1)
                folds.append(self.datas[fold_start:fold_end])
            folds.append(self.datas[(step_size * (k_fold - 1)):])
        if start_fold_t is not None:
            self.fold_t = start_fold_t
        for k in range(self.fold_t, k_fold):
            self.fold_t = k
            self.the_fold_t = self.fold_t
            self.trainset = []
            self.validset = []
            self.testset = []
            if self.read_local_folds:
                self._load_fold_dataset()
            else:
                for fold_t in range(k_fold):
                    if fold_t == k:
                        self.testset += copy.deepcopy(folds[fold_t])
                    else:
                        self.trainset += copy.deepcopy(folds[fold_t])
            self._init_id_from_split()
            parameters = self._preprocess()
            if not self.resume_training and not self.from_pretrained:
                for key, value in parameters.items():
                    setattr(self, key, value)
            parameters = self._build_vocab()
            if not self.resume_training and not self.from_pretrained:
                for key, value in parameters.items():
                    setattr(self, key, value)
            if self.shuffle:
                random.shuffle(self.trainset)
            yield k

    def dataset_load(self):
        r"""dataset process and build vocab.

        when running k-fold setting, this function required to call once per fold.
        """
        if self.k_fold:
            self.the_fold_t +=1
            self.fold_t = self.the_fold_t
            self.testset = []
            self.trainset = []
            self.validset = []
            for fold_t in range(self.k_fold):
                if fold_t == self.the_fold_t:
                    self.testset += copy.deepcopy(self.folds[fold_t])
                else:
                    self.trainset += copy.deepcopy(self.folds[fold_t])
            self._init_id_from_split()

            parameters = self._preprocess()
            if not self.resume_training and not self.from_pretrained:
                for key,value in parameters.items():
                    setattr(self,key,value)
            parameters = self._build_vocab()
            if not self.resume_training and not self.from_pretrained:
                for key, value in parameters.items():
                    setattr(self, key, value)
            if self.resume_training:
                self.resume_training=False

        else:
            parameters = self._preprocess()
            if not self.resume_training and not self.from_pretrained:
                for key, value in parameters.items():
                    setattr(self, key, value)
            parameters = self._build_vocab()
            if not self.resume_training and not self.from_pretrained:
                for key, value in parameters.items():
                    setattr(self, key, value)
            if self.resume_training:
                self.resume_training=False
        if self.shuffle:
            random.shuffle(self.trainset)

    def parameters_to_dict(self):
        """
        return the parameters of dataset as format of dict.
        :return:
        """
        parameters_dict = {}
        for name, value in vars(self).items():
            if hasattr(eval('self.{}'.format(name)), '__call__') or re.match('__.*?__', name):
                continue
            else:
                parameters_dict[name] = copy.deepcopy(value)
        return parameters_dict

    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    def _update_vocab(self, vocab_list):
        raise NotImplementedError

    def save_dataset(self,trained_dir):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError
