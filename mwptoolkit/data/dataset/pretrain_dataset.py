import os
import copy
import warnings
from logging import getLogger

import torch
from transformers import RobertaTokenizer,BertTokenizer

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.enum_type import DatasetName, MaskSymbol, NumMask,TaskType,FixType,Operators,SpecialTokens
from mwptoolkit.utils.preprocess_tools import id_reedit
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_multi_way_tree
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tool.sentence_operator import deprel_tree_to_file, get_group_nums_, span_level_deprel_tree_to_file, get_span_level_deprel_tree_, get_deprel_tree_
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer

class PretrainDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self.task_type=config['task_type']
        self.rule1=config['rule1']
        self.rule2=config['rule2']
        self.pretrained_model_path=config['pretrained_model_path']
        self.embedding=config['embedding']
        self.parse_tree_path = config['parse_tree_file_name']
        if self.parse_tree_path != None:
            self.parse_tree_path = self.dataset_path + '/' + self.parse_tree_path + '.json'
            self.parse_tree_path = os.path.join(self.root,self.parse_tree_path)
    
    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset,self.validset,self.testset = id_reedit(self.trainset, self.validset, self.testset)
        transfer = number_transfer
        
        self.trainset, generate_list, train_copy_nums,unk_symbol = transfer(self.trainset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
        self.validset, _g, valid_copy_nums,_ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
        self.testset, _g, test_copy_nums,_ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
    
        target_equation_fix=self.equation_fix if self.equation_fix else FixType.Infix
        source_equation_fix=self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if self.rule1:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-infix-equation datasets may not surport en rule1 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule1_process(k=max([train_copy_nums, valid_copy_nums, test_copy_nums]))
            else:
                warnings.warn("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")
                #raise Warning("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")

        if self.rule2:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule2_process()
            else:
                warnings.warn("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")
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

        self.generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            self.copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            self.copy_nums = train_copy_nums
        if self.task_type==TaskType.SingleEquation:
            self.operator_list = copy.deepcopy(Operators.Single)
        elif self.task_type==TaskType.MultiEquation:
            self.operator_list = copy.deepcopy(Operators.Multi)
        self.operator_nums = len(self.operator_list)
        
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
        # if self.model.lower() in ["ept"]:
        #     logger = getLogger()
        #     logger.info("build ept information ···")
        #     self.trainset, self.validset, self.testset = \
        #         preprocess_ept_dataset_(self.trainset, self.validset, self.testset, self.dataset)
    
    def _build_vocab(self):
        if self.embedding=='bert':
            tokenizer=BertTokenizer.from_pretrained(self.pretrained_model_path)
        elif self.embedding=='roberta':
            tokenizer=RobertaTokenizer.from_pretrained(self.pretrained_model_path)
        else:
            raise NotImplementedError
        # if self.mask_symbol==MaskSymbol.NUM:
        #     tokenizer.add_tokens(['NUM'])
        # elif self.mask_symbol==MaskSymbol.number:
        #     tokenizer.add_tokens(NumMask.number[:self.copy_nums])
        #tokenizer.special_tokens_map.update({'pad_token':SpecialTokens.PAD_TOKEN})
        global SpecialTokens
        SpecialTokens.PAD_TOKEN=tokenizer.pad_token
        SpecialTokens.SOS_TOKEN=tokenizer.bos_token
        SpecialTokens.EOS_TOKEN=tokenizer.eos_token
        SpecialTokens.UNK_TOKEN=tokenizer.unk_token
        if self.embedding == 'bert':
            SpecialTokens.SOS_TOKEN=tokenizer.cls_token
            SpecialTokens.EOS_TOKEN=tokenizer.sep_token
        self.tokenizer=tokenizer
        self.in_idx2word = list(tokenizer.get_vocab().keys())

        if self.symbol_for_tree:
            self._build_symbol_for_tree()
            self._build_template_symbol_for_tree()
        elif self.equation_fix == FixType.MultiWayTree:
            self._build_symbol_for_multi_way_tree()
            self._build_template_symbol_for_multi_way_tree()
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

    def _build_symbol_for_tree(self):
        self.out_idx2symbol = copy.deepcopy(self.operator_list)
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
        self.out_idx2symbol += self.operator_list
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
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        else:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        if self.model.lower() in ['hms']:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
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
