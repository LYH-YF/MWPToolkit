from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.utils.preprocess_tools import number_transfer,from_infix_to_postfix,from_infix_to_prefix
from mwptoolkit.utils.enum_type import *

class SingleEquationDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self._load_dataset()
        self._preprocess()
        self._build_vocab()

    def _preprocess(self):
        self.trainset,generate_list,copy_nums=number_transfer(self.trainset)
        if self.config["validset_divide"] == True:
            self.validset,_g,_c=number_transfer(self.validset)
        self.testset,_g,_c=number_transfer(self.testset)
        if self.config["equation_fix"]=="prefix":
            fix=from_infix_to_prefix
        elif self.config["equation_fix"]=="postfix":
            fix=from_infix_to_postfix
        else:
            raise ValueError("the equation fix type must be in ['postfix','prefix'].")

        for idx,data in enumerate(self.trainset):
            self.trainset[idx]["equation"]=fix(data["equation"])
        for idx,data in enumerate(self.validset):
            self.validset[idx]["equation"]=fix(data["equation"])
        for idx,data in enumerate(self.testset):
            self.testset[idx]["equation"]=fix(data["equation"])
        
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
        self.out_idx2symbol=OPERATORS
        self.num_start=len(self.out_idx2symbol)

        self.out_idx2symbol+=self.generate_list

        for key,value in words_count.items():
            if value>self.config["min_word_keep"]:
                self.in_idx2word.append(key)
            if key == "NUM":
                self.in_idx2word.append(key)
        
        self.out_idx2symbol+=["N"+str(i) for i in range(self.copy_nums)]
        self.out_idx2symbol+=[UNK_TOKEN]

        self.in_word2idx={}
        self.out_symbol2idx={}
        for idx,word in enumerate(self.in_idx2word):
            self.in_word2idx[word]=idx
        for idx,symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol]=idx

    def get_vocab_size(self):
        return len(self.in_idx2word),len(self.out_idx2symbol)
