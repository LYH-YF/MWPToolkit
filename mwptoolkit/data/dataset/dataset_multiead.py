import copy
import stanza

from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens, FixType, Operators, MaskSymbol, SPECIAL_TOKENS, DatasetName,TaskType
from mwptoolkit.utils.preprocess_tools import number_transfer_math23k, number_transfer_ape200k
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix, from_infix_to_prefix

class DatasetMultiEAD(TemplateDataset):
    def __init__(self, config):
        super().__init__(config)
        self.task_type=config['task_type']
    
    def _preprocess(self):
        if self.dataset == DatasetName.math23k:
            transfer = number_transfer_math23k
        elif self.dataset == DatasetName.ape200k:
            transfer = number_transfer_ape200k
        else:
            NotImplementedError
        self.trainset, generate_list, train_copy_nums = transfer(self.trainset, self.mask_symbol, self.min_generate_keep)
        self.validset, _g, valid_copy_nums = transfer(self.validset, self.mask_symbol, self.min_generate_keep)
        self.testset, _g, test_copy_nums = transfer(self.testset, self.mask_symbol, self.min_generate_keep)
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            self.trainset[idx]["postfix equation"] = from_infix_to_postfix(data["equation"])
            self.trainset[idx]["prefix equation"] = from_infix_to_prefix(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            self.validset[idx]["postfix equation"] = from_infix_to_postfix(data["equation"])
            self.validset[idx]["prefix equation"] = from_infix_to_prefix(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            self.testset[idx]["postfix equation"] = from_infix_to_postfix(data["equation"])
            self.testset[idx]["prefix equation"] = from_infix_to_prefix(data["equation"])
        self.generate_list = generate_list
        if self.symbol_for_tree:
            self.copy_nums=max([train_copy_nums,valid_copy_nums,test_copy_nums])
        else:
            self.copy_nums = train_copy_nums
        
        if self.task_type==TaskType.SingleEquation:
            self.operator_nums = len(Operators.Single)
            self.operator_list = copy.deepcopy(Operators.Single)
        elif self.task_type==TaskType.MultiEquation:
            self.operator_nums = len(Operators.Multi)
            self.operator_list = copy.deepcopy(Operators.Multi)
        else:
            raise NotImplementedError
        self.build_pos()
    
    def _build_vocab(self):
        words_count_1 = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count_1[word] += 1
                except:
                    words_count_1[word] = 1
        self.in_idx2word_1=[SpecialTokens.PAD_TOKEN,SpecialTokens.SOS_TOKEN,SpecialTokens.EOS_TOKEN,SpecialTokens.UNK_TOKEN]
        for key, value in words_count_1.items():
            if value > self.min_word_keep or "NUM" in key:
                self.in_idx2word_1.append(key)
        words_count_2={}
        for data in self.trainset:
            words_list = data["pos"]
            for word in words_list:
                try:
                    words_count_2[word] += 1
                except:
                    words_count_2[word] = 1
        self.in_idx2word_2=[SpecialTokens.UNK_TOKEN]
        for key, value in words_count_2.items():
            if value > self.min_word_keep:
                self.in_idx2word_2.append(key)
        self._build_symbol()
        self._build_symbol_for_tree()

        self.in_word2idx_1 = {}
        self.in_word2idx_2 = {}
        self.out_symbol2idx_1 = {}
        self.out_symbol2idx_2 = {}
        for idx, word in enumerate(self.in_idx2word_1):
            self.in_word2idx_1[word] = idx
        for idx, word in enumerate(self.in_idx2word_2):
            self.in_word2idx_2[word] = idx
        for idx, symbol in enumerate(self.out_idx2symbol_1):
            self.out_symbol2idx_1[symbol] = idx
        for idx, symbol in enumerate(self.out_idx2symbol_2):
            self.out_symbol2idx_2[symbol] = idx

        
        return super()._build_vocab()
    
    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol_1 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Single
        else:
            self.out_idx2symbol_1 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Single
        self.num_start = len(self.out_idx2symbol_1)
        self.out_idx2symbol_1 += self.generate_list
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))
        for data in self.trainset:
            words_list = data["equation"]
            for word in words_list:
                if word in self.out_idx2symbol_1:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    self.out_idx2symbol_1.insert(self.num_start, word)
                    self.num_start += 1
                    continue
                else:
                    self.out_idx2symbol_1.append(word)
        self.out_idx2symbol_1 += [SpecialTokens.UNK_TOKEN]
    
    def _build_symbol_for_tree(self):
        self.out_idx2symbol_2 = copy.deepcopy(Operators.Single)
        self.num_start = len(self.out_idx2symbol_2)
        self.out_idx2symbol_2 += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        self.out_idx2symbol_2 += [SpecialTokens.UNK_TOKEN]
    
    def build_pos(self):
        nlp=stanza.Pipeline(self.language,processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
        for data in self.trainset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos=[]
            parse_tree=[]
            for token in token_list:
                pos.append(token['xpos'])
                parse_tree.append(token['head']-1)
            data['pos']=pos
            data['parse tree']=parse_tree
        for data in self.validset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos=[]
            parse_tree=[]
            for token in token_list:
                pos.append(token['xpos'])
                parse_tree.append(token['head']-1)
            data['pos']=pos
            data['parse tree']=parse_tree
        for data in self.testset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos=[]
            parse_tree=[]
            for token in token_list:
                pos.append(token['xpos'])
                parse_tree.append(token['head']-1)
            data['pos']=pos
            data['parse tree']=parse_tree

