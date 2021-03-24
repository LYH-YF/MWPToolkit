import copy
from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix, from_infix_to_prefix
from mwptoolkit.utils.preprocess_tools import num_transfer_draw, num_transfer_multi, num_transfer_alg514
from mwptoolkit.utils.enum_type import MaskSymbol, Operators, SPECIAL_TOKENS, NumMask, SpecialTokens, FixType, DatasetName

class MultiEquationDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self.equation_fix = config["equation_fix"]
    
    def _preprocess(self):
        if self.dataset==DatasetName.alg514:
            transfer=num_transfer_alg514
        elif self.dataset==DatasetName.draw:
            transfer=num_transfer_draw
        else:
            transfer=num_transfer_multi
        self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(
            self.trainset, self.mask_symbol, self.min_generate_keep,";")
        self.validset, _g, valid_copy_nums, _u = transfer(self.validset,
                                                    self.mask_symbol,
                                                    self.min_generate_keep,";")
        self.testset, _g, test_copy_nums, _u = transfer(self.testset, self.mask_symbol,
                                                    self.min_generate_keep,";")

        if self.equation_fix == FixType.Prefix:
            fix = from_infix_to_prefix
        elif self.equation_fix == FixType.Postfix:
            fix = from_infix_to_postfix
        elif self.equation_fix == FixType.Nonfix:
            fix = None
        else:
            raise NotImplementedError(
                "the type of equation fix ({}) is not implemented.".format(
                    self.equation_fix))

        self.fix_process(fix)
        self.operator_mask_process()

        self.generate_list = unk_symbol+generate_list
        if self.symbol_for_tree:
            self.copy_nums=max([train_copy_nums,valid_copy_nums,test_copy_nums])
        else:
            self.copy_nums = train_copy_nums
        self.operator_nums = len(Operators.Multi)
        self.operator_list = copy.deepcopy(Operators.Multi)
        self.unk_symbol=unk_symbol

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
        self.temp_symbol2idx = {}
        for idx, word in enumerate(self.in_idx2word):
            self.in_word2idx[word] = idx
        for idx, symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(self.temp_idx2symbol):
            self.temp_symbol2idx[symbol] = idx

    def _build_symbol_for_tree(self):
        self.out_idx2symbol = copy.deepcopy(Operators.Multi)
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list
        #self.out_idx2symbol += self.unk_symbol

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem."
                    .format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.generate_list))
        else:
            raise NotImplementedError(
                "the type of masking number ({}) is not implemented".format(
                    self.mask_symbol))

        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        else:
            self.out_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list
        
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem."
                    .format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.generate_list))
        else:
            raise NotImplementedError(
                "the type of masking number ({}) is not implemented".format(
                    self.mask_symbol))

        for data in self.trainset:
            words_list = data["equation"]
            for word in words_list:
                if word in self.out_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    self.out_idx2symbol.insert(self.num_start,word)
                    self.num_start+=1
                    continue
                else:
                    self.out_idx2symbol.append(word)
        self.out_idx2symbol +=[SpecialTokens.UNK_TOKEN]
    
    def _build_template_symbol(self):
        if self.share_vocab:
            self.temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        else:
            self.temp_idx2symbol = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.SOS_TOKEN] + [SpecialTokens.EOS_TOKEN] + Operators.Multi
        
        self.temp_num_start = len(self.temp_idx2symbol)
        self.temp_idx2symbol += self.generate_list
        
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem."
                    .format(self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol += [
                    mask_list[i] for i in range(self.copy_nums)
                ]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(
                        len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError(
                "the type of masking number ({}) is not implemented".format(
                    self.mask_symbol))

        for data in self.trainset:
            words_list = data["template"]
            for word in words_list:
                if word in self.temp_idx2symbol:
                    continue
                elif word[0].isdigit():
                    continue
                elif (word[0].isalpha() or word[0].isdigit()) is not True:
                    self.temp_idx2symbol.insert(self.temp_num_start,word)
                    self.temp_num_start+=1
                    continue
                else:
                    self.temp_idx2symbol.append(word)
        self.temp_idx2symbol +=[SpecialTokens.UNK_TOKEN]
    def get_vocab_size(self):
        return len(self.in_idx2word), len(self.out_idx2symbol)