import copy

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens, FixType, Operators, MaskSymbol, SPECIAL_TOKENS


class TemplateDataset(AbstractDataset):
    """
    you need implement:

    TemplateDataset._preprocess

    TemplateDataset._build_symbol

    TemplateDataset._build_template_symbol
    
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
        raise NotImplementedError

    def _build_template_symbol(self):
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