import os
import copy
import warnings
from logging import getLogger
import re
from collections import Counter

import torch

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix, from_infix_to_prefix, from_infix_to_multi_way_tree, postfix_parser
from mwptoolkit.utils.preprocess_tools import num_transfer_draw, num_transfer_multi, num_transfer_alg514, num_transfer_hmwp
from mwptoolkit.utils.preprocess_tools import deprel_tree_to_file, get_group_nums_, span_level_deprel_tree_to_file, get_span_level_deprel_tree_, get_deprel_tree_, preprocess_ept_dataset_
from mwptoolkit.utils.preprocess_tools import id_reedit,read_aux_jsonl_data
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.enum_type import MaskSymbol, Operators, SPECIAL_TOKENS, NumMask, SpecialTokens, FixType, DatasetName, EPT

from transformers import AutoTokenizer

class MultiEquationDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self.equation_fix = config["equation_fix"]
        self.rule1 = config["rule1"]
        self.rule2 = config["rule2"]
        self.model = config['model']
        self.parse_tree_path = config['parse_tree_file_name']
        if self.parse_tree_path != None:
            self.parse_tree_path = self.dataset_path + '/' + self.parse_tree_path + '.json'
            self.parse_tree_path = os.path.join(self.root,self.parse_tree_path)
        
        if self.model.lower() in ['ept']:
            self.decoder = config["decoder"]
        
        if config["pretrained_model"] != None:
            self.pretrained_model = config["pretrained_model"]
        else:
            self.pretrained_model = None

    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset,self.validset,self.testset = id_reedit(self.trainset, self.validset, self.testset)
        
        if self.dataset == DatasetName.alg514:
            transfer = number_transfer
        elif self.dataset == DatasetName.draw:
            transfer = number_transfer
        elif self.dataset == DatasetName.hmwp:
            transfer = number_transfer
        elif self.dataset == DatasetName.mawps:
            transfer = number_transfer
        else:
            transfer = num_transfer_multi
        if self.dataset in [DatasetName.mawps,DatasetName.alg514,DatasetName.draw,DatasetName.hmwp]:
            self.trainset, generate_list, train_copy_nums,unk_symbol = transfer(self.trainset, self.dataset, 'multi_equation', self.mask_symbol, self.min_generate_keep,";")
            self.validset, _g, valid_copy_nums,_ = transfer(self.validset, self.dataset, 'multi_equation', self.mask_symbol, self.min_generate_keep,";")
            self.testset, _g, test_copy_nums,_ = transfer(self.testset, self.dataset, 'multi_equation', self.mask_symbol, self.min_generate_keep,";")
        else:
            self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(self.trainset, self.mask_symbol, self.min_generate_keep, ";")
            self.validset, _g, valid_copy_nums, _u = transfer(self.validset, self.mask_symbol, self.min_generate_keep, ";")
            self.testset, _g, test_copy_nums, _u = transfer(self.testset, self.mask_symbol, self.min_generate_keep, ";")

        if self.rule1:
            if self.linear and self.single:
                self.en_rule1_process()
            else:
                warnings.warn("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")
                #raise Warning("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")

        if self.rule2:
            if self.linear and self.single:
                self.en_rule2_process()
            else:
                warnings.warn("non-linear or non-single datasets may not surport en rule2 process, already ignored it. ")
                #raise UserWarning("non-linear or non-single datasets may not surport en rule2 process, already ignored it. ")

        if self.equation_fix == FixType.Prefix:
            fix = from_infix_to_prefix
        elif self.equation_fix == FixType.Postfix:
            fix = from_infix_to_postfix
        elif self.equation_fix == FixType.Nonfix:
            fix = None
        elif self.equation_fix == FixType.MultiWayTree:
            fix = from_infix_to_multi_way_tree
        else:
            raise NotImplementedError("the type of equation fix ({}) is not implemented.".format(self.equation_fix))

        self.fix_process(fix)
        self.operator_mask_process()

        self.generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            self.copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            self.copy_nums = train_copy_nums
        self.operator_nums = len(Operators.Multi)
        self.operator_list = copy.deepcopy(Operators.Multi)
        self.unk_symbol = unk_symbol

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
        if self.model.lower() in ['hms']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read span-level deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset, self.max_span_size =\
                    get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build span-level deprel tree infomation to {} ...".format(self.parse_tree_path))
                span_level_deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                                self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset, self.max_span_size =\
                    get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
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
        if self.model.lower() in ["ept"]:
            logger = getLogger()
            logger.info("build ept information ···")
            aux_trainset = []
            aux_testset = []
            
            if self.dataset == DatasetName.alg514:
                for fold_t in range(5):
                    aux_trainset_file = self.dataset_path + "/alg514_fold{}_train.orig.jsonl".format(fold_t)
                    aux_testset_file = self.dataset_path + "/alg514_fold{}_test.orig.jsonl".format(fold_t)
                    aux_trainset += read_aux_jsonl_data(aux_trainset_file)
                    aux_testset += read_aux_jsonl_data(aux_testset_file)
                
                for aux_data in aux_trainset:
                    for dataid, data in enumerate(self.trainset):
                        if data['id'] == int(aux_data["iIndex"]):
                            self.trainset[dataid]["aux"] = aux_data 
                for aux_data in aux_testset:
                    for dataid, data in enumerate(self.testset):
                        if data['id'] == int(aux_data["iIndex"]):
                            self.testset[dataid]["aux"] = aux_data
            if self.dataset == DatasetName.draw:
                aux_trainset_file = self.dataset_path + "/draw_train.orig.jsonl"
                aux_testset_file = self.dataset_path + "/draw_test.orig.jsonl"
                aux_devset_file = self.dataset_path + "/draw_dev.orig.jsonl"
                aux_trainset = read_aux_jsonl_data(aux_trainset_file)
                aux_testset = read_aux_jsonl_data(aux_testset_file)
                aux_devset = read_aux_jsonl_data(aux_devset_file)
                dataset = aux_trainset+aux_testset +aux_devset
                for aux_data in dataset:
                    for dataid, data in enumerate(self.trainset):
                        
                        if data['id'] == aux_data["iIndex"]:
                            self.trainset[dataid]["aux"] = aux_data 
                            
                for aux_data in dataset:
                    for dataid, data in enumerate(self.testset):
                        if data['id'] == aux_data["iIndex"]:
                            self.testset[dataid]["aux"] = aux_data

            if self.dataset == DatasetName.mawps:
                for fold_t in range(5):
                    aux_trainset_file = self.dataset_path + "/mawps_fold{}_train.orig.jsonl".format(fold_t)
                    aux_testset_file = self.dataset_path + "/mawps_fold{}_test.orig.jsonl".format(fold_t)
                    aux_trainset += read_aux_jsonl_data(aux_trainset_file)
                    aux_testset += read_aux_jsonl_data(aux_testset_file)
                
                for aux_data in aux_trainset:
                    for dataid, data in enumerate(self.trainset):
                        if data['original_text'].strip() == aux_data["new_text"].strip():
                            self.trainset[dataid]["aux"] = aux_data 
                for aux_data in aux_testset:
                    for dataid, data in enumerate(self.testset):
                        if data['original_text'].strip() == aux_data["new_text"].strip():
                            self.testset[dataid]["aux"] = aux_data

            
            self.trainset, self.validset, self.testset = \
                preprocess_ept_dataset_(self.trainset, self.validset, self.testset, self.dataset)

    def _build_vocab(self):
        words_count = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count[word] += 1
                except:
                    words_count[word] = 1

        self.in_idx2word = copy.deepcopy(SPECIAL_TOKENS)

        for key, value in words_count.items():
            if value > self.min_word_keep or "NUM" in key:
                self.in_idx2word.append(key)

        if self.pretrained_model:
            pretrained_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            self.in_idx2word = list(pretrained_tokenizer.get_vocab().keys())
            self.in_idx2word.append('[N]')
            for key, value in words_count.items():
                if "N_" in key:
                    self.in_idx2word.append(key)

        if self.symbol_for_tree:
            self._build_symbol_for_tree()
            self._build_template_symbol_for_tree()
        elif self.equation_fix == FixType.MultiWayTree:
            self._build_symbol_for_multi_way_tree()
            self._build_template_symbol_for_multi_way_tree()
        elif self.model.lower() in ["ept"] :
            if 'vall' in self.decoder:
                self._build_symbol_for_ept_op()
            elif 'expr' in self.decoder:
                self._build_symbol_for_ept_expr(self.decoder)
                self.out_opsym2idx = {}
                self.out_consym2idx = {}
                for idx, symbol in enumerate(self.out_idx2opsymbol):
                    self.out_opsym2idx[symbol] = idx
                for idx, symbol in enumerate(self.out_idx2consymbol):
                    self.out_consym2idx[symbol] = idx
            self._build_template_symbol()
        else:
            self._build_symbol()
            self._build_template_symbol()

        # if self.share_vocab:
        #     for symbol in self.out_idx2symbol:
        #         if symbol in self.in_idx2word:
        #             continue
        #         else:
        #             self.in_idx2word.append(symbol)
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
        self.out_idx2symbol = special_tokens
        #equation_counter.update(special_tokens)
        for token in list(equation_counter.keys()):
            if token not in self.out_idx2symbol:
                self.out_idx2symbol.append(token)
    
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
        self.out_idx2opsymbol = EPT.FUN_TOKENS_WITH_EQ.copy()
        self.out_idx2consymbol = constant_specials
        for token in list(operator_counter.keys()):
            if token not in self.out_idx2opsymbol:
                self.out_idx2opsymbol.append(token)
        for token in list(constant_counter.keys()):
            if token not in self.out_idx2consymbol:
                self.out_idx2consymbol.append(token)

        #operator_counter.update(EPT.FUN_TOKENS_WITH_EQ.copy())
        #constant_counter.update(constant_specials)
        #self.out_idx2opsymbol = list(operator_counter.keys())
        #self.out_idx2consymbol = list(constant_counter.keys())
        self.out_idx2symbol = self.out_idx2consymbol+self.out_idx2opsymbol


    def _build_symbol_for_tree(self):
        self.out_idx2symbol = copy.deepcopy(Operators.Multi)
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_symbol_for_multi_way_tree(self):
        self.out_idx2symbol = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.NON_TOKEN]
        self.out_idx2symbol += Operators.Multi
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))
        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        else:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        if self.model.lower() in ['hms']:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list
        if self.model.lower() in ['hms']:
            self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        for data in self.trainset:
            words_list = data["equation"]
            for word in words_list:
                if word in self.out_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    self.out_idx2symbol.insert(self.num_start, word)
                    self.num_start += 1
                    continue
                else:
                    self.out_idx2symbol.append(word)
        if self.model.lower() in ['hms']:
            return
        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_template_symbol_for_multi_way_tree(self):
        self.temp_idx2symbol = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.NON_TOKEN, SpecialTokens.OPT_TOKEN]
        self.temp_num_start = len(self.temp_idx2symbol)
        self.temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        self.temp_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_template_symbol(self):
        if self.share_vocab:
            self.temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + [SpecialTokens.OPT_TOKEN]
        else:
            self.temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + [SpecialTokens.OPT_TOKEN]

        self.temp_num_start = len(self.temp_idx2symbol)
        self.temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        for data in self.trainset:
            words_list = data["template"]
            for word in words_list:
                if word in self.temp_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    self.temp_idx2symbol.insert(self.temp_num_start, word)
                    self.temp_num_start += 1
                    continue
                else:
                    self.temp_idx2symbol.append(word)
        self.temp_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_template_symbol_for_tree(self):
        self.temp_idx2symbol = [SpecialTokens.OPT_TOKEN]
        self.temp_num_start = len(self.temp_idx2symbol)
        self.temp_idx2symbol += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        self.temp_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _update_vocab(self, vocab_list):
        index = len(self.in_idx2word)
        for word in vocab_list:
            if word not in self.in_idx2word:
                self.in_idx2word.append(word)
                self.in_word2idx[word] = index
                index += 1

    def get_vocab_size(self):
        return len(self.in_idx2word), len(self.out_idx2symbol)