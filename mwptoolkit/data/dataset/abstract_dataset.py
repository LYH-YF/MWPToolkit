import random
import copy
import torch
from mwptoolkit.utils.utils import read_json_data,write_json_data
from mwptoolkit.utils.preprocess_tools import operator_mask,EN_rule1_stat,EN_rule2_
from mwptoolkit.utils.preprocess_tools import get_group_nums,get_deprel_tree,get_span_level_deprel_tree

class AbstractDataset(object):
    '''abstract dataset'''
    def __init__(self, config):
        super().__init__()
        self.validset_divide = config["validset_divide"]
        self.dataset_path = config["dataset_path"]
        self.min_word_keep = config["min_word_keep"]
        self.min_generate_keep = config["min_generate_keep"]
        self.mask_symbol = config["mask_symbol"]
        self.symbol_for_tree = config["symbol_for_tree"]
        self.share_vocab = config["share_vocab"]
        self.k_fold = config["k_fold"]
        self.dataset = config["dataset"]
        self.model = config["model"]
        self.read_local_folds = config["read_local_folds"]
        self.language=config["language"]
        self.single = config["single"]
        self.linear = config["linear"]
        self.device = config["device"]
        self.equation_fix = config["equation_fix"]
        self.max_span_size = 1

    def _load_dataset(self):
        '''
        read dataset from files
        '''
        trainset_file = self.dataset_path + "/trainset.json"
        validset_file = self.dataset_path + "/validset.json"
        testset_file = self.dataset_path + "/testset.json"
        self.trainset = read_json_data(trainset_file)[:10]
        self.validset = read_json_data(validset_file)[:10]
        self.testset = read_json_data(testset_file)[:10]
    def _load_fold_dataset(self):
        trainset_file = self.dataset_path + "/trainset_fold{}.json".format(self.fold_t)
        testset_file = self.dataset_path + "/testset_fold{}.json".format(self.fold_t)
        self.trainset = read_json_data(trainset_file)
        self.testset = read_json_data(testset_file)
        self.validset = []
    def fix_process(self, fix):
        r"""equation process

        Args:
            fix: a function to make postfix, prefix or None  
        """
        if fix != None:
            for idx, data in enumerate(self.trainset):
                self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.trainset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.validset):
                self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.validset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.testset):
                self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.testset[idx]["equation"] = fix(data["equation"])
        else:
            for idx, data in enumerate(self.trainset):
                self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.validset):
                self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.testset):
                self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                
    def operator_mask_process(self):
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["template"] = operator_mask(data["equation"])
    
    def en_rule1_process(self,k):
        rule1_list=EN_rule1_stat(self.trainset,k)
        for idx, data in enumerate(self.trainset):
            flag=False
            equ_list=data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.trainset[idx]["equation"] = equ_lists[0]
                    flag=True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.validset):
            flag=False
            equ_list=data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.validset[idx]["equation"] = equ_lists[0]
                    flag=True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.testset):
            flag=False
            equ_list=data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.testset[idx]["equation"] = equ_lists[0]
                    flag=True
                    break
                if flag:
                    break
    
    def en_rule2_process(self):
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["equation"] = EN_rule2_(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["equation"] = EN_rule2_(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["equation"] = EN_rule2_(data["equation"])
    
    def build_group_nums_for_graph(self):
        use_gpu=True if self.device==torch.device('cuda') else False
        self.trainset=get_group_nums(self.trainset,self.language,use_gpu)
        self.validset=get_group_nums(self.validset,self.language,use_gpu)
        self.testset=get_group_nums(self.testset,self.language,use_gpu)
    
    def build_deprel_tree(self):
        self.trainset,tokens=get_deprel_tree(self.trainset,self.language)
        self.validset,_=get_deprel_tree(self.validset,self.language)
        self.testset,_=get_deprel_tree(self.testset,self.language)

        #self._update_vocab(tokens)
    
    def build_span_level_deprel_tree(self):
        self.trainset,train_span_szie=get_span_level_deprel_tree(self.trainset,self.language)
        self.validset,valid_span_size=get_span_level_deprel_tree(self.validset,self.language)
        self.testset,test_span_size=get_span_level_deprel_tree(self.testset,self.language)
        self.max_span_size=max([train_span_szie,valid_span_size,test_span_size])
    
    def cross_validation_load(self, k_fold, start_fold_t=0):
        r"""dataset load for cross validation

        Build folds for cross validation.Choose one of folds divided into validset and testset and other folds for trainset.
        
        Args:
            k_fold: int, the number of folds, also the cross validation parameter k.
            start_fold_t: int|defalte 0, training start from the training of t-th time.
        Return:
            Generator including current training index of cross validation.
        """
        if k_fold == 0 or k_fold == 1:
            raise ValueError("the cross validation parameter k shouldn't be zero or one, it should be greater than one")
        if self.read_local_folds !=True:
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
        self.start_fold_t = start_fold_t
        for k in range(self.start_fold_t, k_fold):
            self.fold_t=k
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
            self._preprocess()
            self._build_vocab()
            yield k

    def dataset_load(self):
        r"""dataset process and build vocab
        """
        self._load_dataset()
        self._preprocess()
        self._build_vocab()

    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError
    
    def _update_vocab(self,vocab_list):
        raise NotImplementedError

