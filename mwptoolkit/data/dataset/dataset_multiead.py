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

