# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:00
# @File: dataset_ept.py
import json
import os
import re
import copy
from logging import getLogger
from collections import Counter

import torch
import stanza
from transformers import AutoTokenizer, AlbertTokenizer, BertTokenizer

from mwptoolkit.config.configuration import Config
from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, \
    from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tool.equation_operator import postfix_parser
from mwptoolkit.utils.preprocess_tool.dataset_operator import preprocess_ept_dataset_
from mwptoolkit.utils.preprocess_tools import id_reedit, read_aux_jsonl_data
from mwptoolkit.utils.enum_type import MaskSymbol, Operators, SPECIAL_TOKENS, NumMask, SpecialTokens, FixType, \
    DatasetName, EPT
from mwptoolkit.utils.utils import read_json_data, write_json_data


class DatasetEPT(TemplateDataset):
    """dataset class for deep-learning model EPT.
    """

    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

        task_type (str): [single_equation | multi_equation], the type of task.

        pretrained_model or transformers_pretrained_model (str|None): road path or name of pretrained model.

        decoder (str): decoder module name.

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
        self.pretrained_model = config['pretrained_model_path']
        self.decoder = config['decoder']
        self.task_type = config['task_type']

    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset, self.validset, self.testset = id_reedit(self.trainset, self.validset, self.testset)

        transfer = number_transfer

        self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(self.trainset, self.dataset,
                                                                             self.task_type, self.mask_symbol,
                                                                             self.min_generate_keep,self.linear, ";")
        self.validset, _g, valid_copy_nums, _ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol,
                                                         self.min_generate_keep,self.linear, ";")
        self.testset, _g, test_copy_nums, _ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol,
                                                       self.min_generate_keep,self.linear, ";")

        source_equation_fix = self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if source_equation_fix == FixType.Infix:
            fix = from_infix_to_postfix
        elif source_equation_fix == FixType.Prefix:
            fix = from_prefix_to_postfix
        elif source_equation_fix == FixType.Postfix:
            fix = None
        else:
            raise NotImplementedError()
        self.fix_process(fix)
        self.operator_mask_process()

        generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            copy_nums = train_copy_nums
        operator_nums = len(Operators.Multi)
        operator_list = copy.deepcopy(Operators.Multi)

        logger = getLogger()
        logger.info("build ept information ···")
        aux_trainset = []
        aux_testset = []

        if self.dataset == DatasetName.alg514:
            for fold_t in range(1):
                aux_trainset_file = self.dataset_path + "/alg514_fold{}_train.orig.jsonl".format(fold_t)
                aux_testset_file = self.dataset_path + "/alg514_fold{}_test.orig.jsonl".format(fold_t)
                if not os.path.isabs(aux_trainset_file):
                    aux_trainset_file = os.path.join(os.getcwd(), aux_trainset_file)
                if not os.path.isabs(aux_testset_file):
                    aux_testset_file = os.path.join(os.getcwd(), aux_testset_file)

                aux_trainset += read_aux_jsonl_data(aux_trainset_file)
                aux_testset += read_aux_jsonl_data(aux_testset_file)
            dataset = aux_trainset + aux_testset

            for dataid, data in enumerate(self.trainset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.trainset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.validset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.validset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.testset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.testset[dataid]["aux"] = aux_data
                        break
        if self.dataset == DatasetName.draw:
            aux_trainset_file = self.dataset_path + "/draw_train.orig.jsonl"
            aux_testset_file = self.dataset_path + "/draw_test.orig.jsonl"
            aux_devset_file = self.dataset_path + "/draw_dev.orig.jsonl"
            if not os.path.isabs(aux_trainset_file):
                aux_trainset_file = os.path.join(os.getcwd(), aux_trainset_file)
            if not os.path.isabs(aux_testset_file):
                aux_testset_file = os.path.join(os.getcwd(), aux_testset_file)
            if not os.path.isabs(aux_devset_file):
                aux_devset_file = os.path.join(os.getcwd(), aux_devset_file)

            aux_trainset = read_aux_jsonl_data(aux_trainset_file)
            aux_testset = read_aux_jsonl_data(aux_testset_file)
            aux_devset = read_aux_jsonl_data(aux_devset_file)
            dataset = aux_trainset + aux_testset + aux_devset

            for dataid, data in enumerate(self.trainset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.trainset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.validset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.validset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.testset):
                for aux_data in dataset:
                    if data['id'] == int(aux_data["iIndex"]):
                        self.testset[dataid]["aux"] = aux_data
                        break

        if self.dataset == DatasetName.mawps:
            for fold_t in range(1):
                aux_trainset_file = self.dataset_path + "/mawps_fold{}_train.orig.jsonl".format(fold_t)
                aux_testset_file = self.dataset_path + "/mawps_fold{}_test.orig.jsonl".format(fold_t)
                if not os.path.isabs(aux_trainset_file):
                    aux_trainset_file = os.path.join(os.getcwd(), aux_trainset_file)
                if not os.path.isabs(aux_testset_file):
                    aux_testset_file = os.path.join(os.getcwd(), aux_testset_file)

                aux_trainset += read_aux_jsonl_data(aux_trainset_file)
                aux_testset += read_aux_jsonl_data(aux_testset_file)
            dataset = aux_trainset + aux_testset

            for dataid, data in enumerate(self.trainset):
                for aux_data in dataset:
                    if data['original_text'].strip() == aux_data["new_text"].strip():
                        self.trainset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.validset):
                for aux_data in dataset:
                    if data['original_text'].strip() == aux_data["new_text"].strip():
                        self.validset[dataid]["aux"] = aux_data
                        break
            for dataid, data in enumerate(self.testset):
                for aux_data in dataset:
                    if data['original_text'].strip() == aux_data["new_text"].strip():
                        self.testset[dataid]["aux"] = aux_data
                        break

        self.trainset, self.validset, self.testset = \
            preprocess_ept_dataset_(self.trainset, self.validset, self.testset, self.dataset)

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

        in_idx2word = copy.deepcopy(SPECIAL_TOKENS)

        for key, value in words_count.items():
            if value > self.min_word_keep or "NUM" in key:
                in_idx2word.append(key)

        if self.pretrained_model:
            if self.dataset in ['math23k', 'hmwp']:
                pretrained_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
            else:
                pretrained_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            in_idx2word = list(pretrained_tokenizer.get_vocab().keys())
            in_idx2word.append('[N]')
            for key, value in words_count.items():
                if "N_" in key:
                    in_idx2word.append(key)

        if 'vall' in self.decoder:
            equ_dict = self._build_symbol_for_ept_op()
            out_idx2symbol = equ_dict['out_idx2symbol']
        elif 'expr' in self.decoder:
            equ_dict = self._build_symbol_for_ept_expr(self.decoder)
            out_idx2consymbol = equ_dict['out_idx2consymbol']
            out_idx2opsymbol = equ_dict['out_idx2opsymbol']
            out_idx2symbol = equ_dict['out_idx2symbol']

            out_opsym2idx = {}
            out_consym2idx = {}
            for idx, symbol in enumerate(out_idx2opsymbol):
                out_opsym2idx[symbol] = idx
            for idx, symbol in enumerate(out_idx2consymbol):
                out_consym2idx[symbol] = idx
        else:
            raise NotImplementedError
        temp_dict = self._build_template_symbol()
        temp_idx2symbol = temp_dict['temp_idx2symbol']
        temp_num_start = temp_dict['temp_num_start']

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
                       'temp_symbol2idx': temp_symbol2idx,
                       'temp_num_start': temp_num_start}
        return_info.update(equ_dict)
        return return_info

    def _build_symbol_for_ept_op(self):
        def preprocess(formulae):
            """
            Tokenize equation using Op tokens.

            :param List[Tuple[int,str]] formulae:
                List of equations. Each equation is a tuple of following.
                - [0] Indicates type of equation (0: equation, 1: answer tuple, and 2: memory)
                - [1] String of expression
            :rtype: List[str]
            :return: List of Op tokens.
            """
            assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                           "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

            tokens = []
            memory_counter = 0
            variables = {}

            for typ, expr in formulae:
                if type(expr) is str:
                    expr = re.split('\\s+', expr.strip())

                if typ == EPT.PREP_KEY_ANS:
                    # Ignore answer tuple
                    continue
                elif typ == EPT.PREP_KEY_MEM:
                    # If this is a memory, then make it as M_<id> = <expr>.
                    expr = ['M_%s' % memory_counter] + expr + ['=']
                    memory_counter += 1

                for token in expr:
                    # Normalize tokens
                    if any(token.startswith(prefix) for prefix in ['X_']):
                        # Remapping variables by order of appearance.
                        if token not in variables:
                            variables[token] = len(variables)

                        position = variables[token]
                        token = EPT.FORMAT_VAR % position  # By the index of the first appearance.
                        tokens.append(token)
                    elif any(token.startswith(prefix) for prefix in ['N_']):
                        # To preserve order, we padded indices with zeros at the front.
                        position = int(token.split('_')[-1])
                        tokens.append(EPT.FORMAT_NUM % position)
                    else:
                        if token.startswith('C_'):
                            token = token.replace('C_', EPT.CON_PREFIX)
                        tokens.append(token)

            return tokens

        equation_counter = Counter()
        for data in self.trainset:
            words_list = data["ept"]["expr"]
            equation_counter.update([tok for tok in preprocess(words_list) if tok != -1])
        special_tokens = EPT.SEQ_TOKENS.copy()

        special_tokens += [EPT.FORMAT_NUM % i for i in range(EPT.NUM_MAX)]
        special_tokens += [EPT.FORMAT_VAR % i for i in range(EPT.VAR_MAX)]
        out_idx2symbol = special_tokens
        # equation_counter.update(special_tokens)
        for token in list(equation_counter.keys()):
            if token not in self.out_idx2symbol:
                out_idx2symbol.append(token)
        return {'out_idx2symbol':out_idx2symbol}

    def _build_symbol_for_ept_expr(self, decoder_type):
        def preprocess(formulae):
            """
            Tokenize equation using Op tokens.

            :param List[Tuple[int,str]] formulae:
                List of equations. Each equation is a tuple of following.
                - [0] Indicates type of equation (0: equation, 1: answer tuple, and 2: memory)
                - [1] String of expression
            :rtype: List[str]
            :return: List of Op tokens.
            """
            assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                           "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

            variables = []
            memories = []

            for typ, expr in formulae:
                if type(expr) is str:
                    expr = re.split('\\s+', expr.strip())

                # Replace number, const, variable tokens with N_<id>, C_<value>, X_<id>
                normalized = []
                for token in expr:
                    if any(token.startswith(prefix) for prefix in ['X_']):
                        # Case 1: Variable
                        if token not in variables:
                            variables.append(token)

                        # Set as negative numbers, since we don't know how many variables are in the list.
                        normalized.append((EPT.ARG_MEM, - variables.index(token) - 1))
                    elif any(token.startswith(prefix) for prefix in ['N_']):
                        # Case 2: Number
                        token = int(token.split('_')[-1])
                        if 'gen' in decoder_type:
                            # Treat number indicator as constant.
                            normalized.append((EPT.ARG_NUM, EPT.FORMAT_NUM % token))
                        else:
                            normalized.append((EPT.ARG_NUM, token))
                    elif token.startswith('C_'):
                        normalized.append((EPT.ARG_CON, token.replace('C_', EPT.CON_PREFIX)))
                    else:
                        normalized.append(token)

                # Build expressions (ignore answer tuples)
                if typ == EPT.PREP_KEY_EQN:
                    stack_len = postfix_parser(normalized, memories)
                    assert stack_len == 1, "Equation is not correct! '%s'" % expr
                elif typ == EPT.PREP_KEY_MEM:
                    stack_len = postfix_parser(normalized, memories)
                    assert stack_len == 1, "Intermediate representation of memory is not correct! '%s'" % expr

            # Reconstruct indices for result of prior expression.
            var_length = len(variables)
            # Add __NEW_VAR at the front of the sequence. The number of __NEW_VAR()s equals to the number of variables used.
            preprocessed = [(EPT.FUN_NEW_VAR, []) for _ in range(var_length)]
            for operator, operands in memories:
                # For each expression
                new_arguments = []
                for typ, tok in operands:
                    if typ == EPT.ARG_MEM:
                        # Shift index of prior expression by the number of variables.
                        tok = tok + var_length if tok >= 0 else -(tok + 1)

                        if 'gen' in decoder_type:
                            # Build as a string
                            tok = EPT.FORMAT_MEM % tok

                    new_arguments.append((typ, tok))

                # Register an expression
                preprocessed.append((operator, new_arguments))

            return preprocessed

        operator_counter = Counter()
        constant_counter = Counter()

        constant_specials = [EPT.ARG_UNK]
        if 'gen' in decoder_type:
            # Enforce index of numbers become 1 ~ NUM_MAX
            constant_specials += [EPT.FORMAT_NUM % i for i in range(EPT.NUM_MAX)]
            # Enforce index of memory indices become NUM_MAX+1 ~ NUM_MAX+MEM_MAX
            constant_specials += [EPT.FORMAT_MEM % i for i in range(EPT.MEM_MAX)]

        for data in self.trainset:
            # Equation is not tokenized

            item = preprocess(data['ept']['expr'])
            # Count operators
            operator, operands = zip(*item)
            operator_counter.update(operator)
            for operand in operands:
                # Count constant operands (all operands if self.force_generation)
                constant_counter.update([const for t, const in operand if t == EPT.ARG_CON or 'gen' in decoder_type])
        out_idx2opsymbol = EPT.FUN_TOKENS_WITH_EQ.copy()
        out_idx2consymbol = constant_specials
        for token in list(operator_counter.keys()):
            if token not in out_idx2opsymbol:
                out_idx2opsymbol.append(token)
        for token in list(constant_counter.keys()):
            if token not in out_idx2consymbol:
                out_idx2consymbol.append(token)
        out_idx2symbol = out_idx2consymbol + out_idx2opsymbol

        return {'out_idx2consymbol':out_idx2consymbol,'out_idx2opsymbol':out_idx2opsymbol,'out_idx2symbol':out_idx2symbol}

        # operator_counter.update(EPT.FUN_TOKENS_WITH_EQ.copy())
        # constant_counter.update(constant_specials)
        # self.out_idx2opsymbol = list(operator_counter.keys())
        # self.out_idx2consymbol = list(constant_counter.keys())

    def _build_template_symbol(self):
        if self.share_vocab:
            temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + [SpecialTokens.OPT_TOKEN]
        else:
            temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + [
                SpecialTokens.OPT_TOKEN]

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
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(
                        self.copy_nums))
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

    def save_dataset(self, save_dir: str):
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
                'trainset_id': self.trainset_id,
                'validset_id': self.validset_id,
                'testset_id': self.testset_id,
                'folds_id': self.folds_id
            },
            data_id_file
        )
        json_encoder = json.encoder.JSONEncoder()
        parameters_dict = self.parameters_to_dict()
        not_support_json=[]
        not_to_save = ['in_idx2word', 'out_idx2symbol', 'temp_idx2symbol', 'in_word2idx', 'out_symbol2idx',
                       'temp_symbol2idx', 'folds', 'trainset', 'testset', 'validset', 'datas', 'trainset_id',
                       'validset_id', 'testset_id', 'folds_id']
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
    def load_from_pretrained(cls, pretrained_dir: str, resume_training=False):
        """
        load dataset parameters from file.

        :param pretrained_dir: (str) folder which saved the parameter file
        :param resume_training: (bool) load parameter for resuming training or not.
        :return: an instantiated object
        """
        config = Config.load_from_pretrained(pretrained_dir)
        dataset = DatasetEPT(config)

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

        setattr(self,'in_idx2word',in_idx2word)
        setattr(self, 'out_idx2symbol', out_idx2symbol)
        setattr(self, 'temp_idx2symbol', temp_idx2symbol)
        setattr(self, 'in_word2idx', in_word2idx)
        setattr(self, 'out_symbol2idx', out_symbol2idx)
        setattr(self, 'temp_symbol2idx', temp_symbol2idx)
        for key,value in parameter_dict.items():
            setattr(self,key,value)
