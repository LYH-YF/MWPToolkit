import copy
from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix, from_infix_to_prefix
from mwptoolkit.utils.preprocess_tools import number_transfer_, number_transfer_ape200k
from mwptoolkit.utils.enum_type import MaskSymbol, NumMask, SpecialTokens, FixType, Operators, DatasetName
from mwptoolkit.utils.enum_type import OPERATORS, SPECIAL_TOKENS

class SingleEquationDataset(AbstractDataset):
    def __init__(self, config):
        self.equation_fix = config["equation_fix"]
        super().__init__(config)

    def _preprocess(self):
        if self.dataset == DatasetName.math23k:
            transfer = number_transfer_
        elif self.dataset == DatasetName.ape200k:
            transfer = number_transfer_ape200k
        else:
            NotImplementedError
        self.trainset, generate_list, train_copy_nums = transfer(self.trainset, self.mask_symbol, self.min_generate_keep)
        self.validset, _g, valid_copy_nums = transfer(self.validset, self.mask_symbol, self.min_generate_keep)
        self.testset, _g, test_copy_nums = transfer(self.testset, self.mask_symbol, self.min_generate_keep)

        if self.equation_fix == FixType.Prefix:
            fix = from_infix_to_prefix
        elif self.equation_fix == FixType.Postfix:
            fix = from_infix_to_postfix
        elif self.equation_fix == FixType.Nonfix:
            fix = None
        else:
            raise NotImplementedError("the type of equation fix ({}) is not implemented.".format(self.equation_fix))

        self.fix_process(fix)

        self.generate_list = generate_list
        self.copy_nums = train_copy_nums
        self.operator_nums = len(Operators.Single)
        self.operator_list = copy.deepcopy(Operators.Single)

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

        if self.symbol_for_tree:
            self._build_symbol_for_tree()
        else:
            self._build_symbol()

        if self.share_vocab:
            for symbol in self.out_idx2symbol:
                if symbol in self.in_idx2word:
                    continue
                else:
                    self.in_idx2word.append(symbol)

        self.in_word2idx = {}
        self.out_symbol2idx = {}
        for idx, word in enumerate(self.in_idx2word):
            self.in_word2idx[word] = idx
        for idx, symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol] = idx

    def _build_symbol_for_tree(self):
        self.out_idx2symbol = copy.deepcopy(Operators.Single)
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

    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + OPERATORS
        else:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + OPERATORS

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
        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def get_vocab_size(self):
        return len(self.in_idx2word), len(self.out_idx2symbol)
