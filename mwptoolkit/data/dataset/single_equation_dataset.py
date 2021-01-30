from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import number_transfer_,from_infix_to_postfix,from_infix_to_prefix
from mwptoolkit.utils.enum_type import *

class SingleEquationDataset(AbstractDataset):
    def __init__(self, config):
        self.equation_fix=config["equation_fix"]
        super().__init__(config)
        self._load_dataset()
        self._preprocess()
        self._build_vocab()

    def _preprocess(self):
        self.trainset,generate_list,copy_nums=number_transfer_(self.trainset,self.mask_symbol)
        if self.validset_divide == True:
            self.validset,_g,_c=number_transfer_(self.validset,self.mask_symbol)
        self.testset,_g,_c=number_transfer_(self.testset,self.mask_symbol)
        
        if self.equation_fix=="prefix":
            fix=from_infix_to_prefix
        elif self.equation_fix=="postfix":
            fix=from_infix_to_postfix
        else:
            fix=None

        self.fix_process(fix)
        
        self.generate_list=generate_list
        self.copy_nums=copy_nums
        self.operator_nums=len(OPERATORS)
    
    def _build_vocab(self):
        words_count={}
        for data in self.trainset:
            words_list=data["question"]
            for word in words_list:
                try:
                    words_count[word]+=1
                except:
                    words_count[word]=0
        
        self.in_idx2word=INPUT_SPECIAL_TOKENS

        for key,value in words_count.items():
            if value>self.min_word_keep:
                self.in_idx2word.append(key)
            if "NUM" in key:
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

        self.in_word2idx={}
        self.out_symbol2idx={}
        for idx,word in enumerate(self.in_idx2word):
            self.in_word2idx[word]=idx
        for idx,symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol]=idx
    def _build_symbol_for_tree(self):
        self.out_idx2symbol=OPERATORS
        self.num_start=len(self.out_idx2symbol)
        self.out_idx2symbol+=self.generate_list

        alphabet="abcdefghijklmnopqrstuvwxyz"
        self.out_idx2symbol+=["NUM_"+alphabet[i] for i in range(self.copy_nums)]
        self.out_idx2symbol+=[UNK_TOKEN]
    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol=[PAD_TOKEN]+[EOS_TOKEN]+OPERATORS
        else:
            self.out_idx2symbol=[PAD_TOKEN]+[EOS_TOKEN]+[SOS_TOKEN]+OPERATORS
        
        self.num_start=len(self.out_idx2symbol)
        self.out_idx2symbol+=self.generate_list
        for data in self.trainset:
            words_list=data["equation"]
            for word in words_list:
                if word in self.out_idx2symbol:
                    continue
                else:
                    self.out_idx2symbol.append(word)
        #self.out_idx2symbol+=[UNK_TOKEN]
    def get_vocab_size(self):
        return len(self.in_idx2word),len(self.out_idx2symbol)
