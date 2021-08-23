# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:11
# @File: dataset_multiencdec.py


import os
import copy
from logging import getLogger
import stanza

from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens, FixType, Operators, MaskSymbol, SPECIAL_TOKENS, DatasetName, TaskType
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tools import id_reedit,dataset_drop_duplication
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.utils import read_json_data,write_json_data

class DatasetMultiEncDec(TemplateDataset):
    """dataset class for deep-learning model MultiE&D
    """
    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

        task_type (str): [single_equation | multi_equation], the type of task.

        parse_tree_file_name (str|None): the name of the file to save parse tree infomation.

        ltp_model_path (str|None): the road path of ltp model.

        model (str): model name.

        dataset (str): dataset name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.
        
        dataset_path (str): the road path of dataset folder.

        language (str): a property of dataset, the language of dataset.

        single (bool): a property of dataset, the equation of dataset is single or not.

        linear (bool): a property of dataset, the equation of dataset is linear or not.

        source_equation_fix (str): [infix | postfix | prefix], a property of dataset, the source format of equation of dataset.

        rebuild (bool): when loading additional dataset infomation, this can decide to build infomation anew or load infomation built before.

        validset_divide (bool): whether to split validset. if True, the dataset is split to trainset-validset-testset. if False, the dataset is split to trainset-testset.

        mask_symbol (str): [NUM | number], the symbol to mask numbers in equation.
        
        min_word_keep (int): in dataset, words that count greater than the value, will be kept in input vocabulary.
        
        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        k_fold (int|None): if it's an integer, it indicates to run k-fold cross validation. if it's None, it indicates to run trainset-validset-testset split.

        read_local_folds (bool): when running k-fold cross validation, if True, then loading split folds from dataset folder. if False, randomly split folds.

        """
        super().__init__(config)
        self.task_type = config['task_type']
        self.parse_tree_path = config['parse_tree_file_name']
        if self.parse_tree_path != None:
            self.parse_tree_path = self.dataset_path+'/'+self.parse_tree_path+'.json'
            self.parse_tree_path = os.path.join(self.root,self.parse_tree_path)
        self.ltp_model_path=config['ltp_model_path']
        if self.ltp_model_path and not os.path.isabs(self.ltp_model_path):
            self.ltp_model_path = os.path.join(self.root,self.ltp_model_path)
    
    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset,self.validset,self.testset = id_reedit(self.trainset, self.validset, self.testset)
        if self.dataset in [DatasetName.draw]:
            self.trainset,self.validset,self.testset = dataset_drop_duplication(self.trainset, self.validset, self.testset)
        transfer = number_transfer
        
        self.trainset, generate_list, train_copy_nums,unk_symbol = transfer(self.trainset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
        self.validset, _g, valid_copy_nums,_ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
        self.testset, _g, test_copy_nums,_ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol, self.min_generate_keep,";")
        source_equation_fix=self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if source_equation_fix==FixType.Infix:
            to_infix=None
            to_prefix=from_infix_to_prefix
            to_postfix=from_infix_to_postfix
        elif source_equation_fix==FixType.Prefix:
            to_infix=from_prefix_to_infix
            to_prefix=None
            to_postfix=from_prefix_to_postfix
        elif source_equation_fix==FixType.Postfix:
            to_infix=from_postfix_to_infix
            to_prefix=from_postfix_to_prefix
            to_postfix=None
        else:
            raise NotImplementedError()
        for idx, data in enumerate(self.trainset):
            if to_infix:
                self.trainset[idx]["infix equation"] = to_infix(data["equation"])
            else:
                self.trainset[idx]["infix equation"] = data["equation"]
            if to_postfix:
                self.trainset[idx]["postfix equation"] = to_postfix(data["equation"])
            else:
                self.trainset[idx]["postfix equation"] = data["equation"]
            if to_prefix:
                self.trainset[idx]["prefix equation"] = to_prefix(data["equation"])
            else:
                self.trainset[idx]["prefix equation"] = data["equation"]
        for idx, data in enumerate(self.validset):
            if to_infix:
                self.validset[idx]["infix equation"] = to_infix(data["equation"])
            else:
                self.validset[idx]["infix equation"] = data["equation"]
            if to_postfix:
                self.validset[idx]["postfix equation"] = to_postfix(data["equation"])
            else:
                self.validset[idx]["postfix equation"] = data["equation"]
            if to_prefix:
                self.validset[idx]["prefix equation"] = to_prefix(data["equation"])
            else:
                self.validset[idx]["prefix equation"] = data["equation"]
        for idx, data in enumerate(self.testset):
            if to_infix:
                self.testset[idx]["infix equation"] = to_infix(data["equation"])
            else:
                self.testset[idx]["infix equation"] = data["equation"]
            if to_postfix:
                self.testset[idx]["postfix equation"] = to_postfix(data["equation"])
            else:
                self.testset[idx]["postfix equation"] = data["equation"]
            if to_prefix:
                self.testset[idx]["prefix equation"] = to_prefix(data["equation"])
            else:
                self.testset[idx]["prefix equation"] = data["equation"]
        
        self.generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            self.copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            self.copy_nums = train_copy_nums

        if self.task_type == TaskType.SingleEquation:
            self.operator_nums = len(Operators.Single)
            self.operator_list = copy.deepcopy(Operators.Single)
        elif self.task_type == TaskType.MultiEquation:
            self.operator_nums = len(Operators.Multi)
            self.operator_list = copy.deepcopy(Operators.Multi)
        else:
            raise NotImplementedError
        if os.path.exists(self.parse_tree_path) and not self.rebuild:
            logger = getLogger()
            logger.info('read pos infomation from {} ...'.format(self.parse_tree_path))
            self.read_pos_from_file(self.parse_tree_path)
        else:
            logger = getLogger()
            logger.info('build pos infomation to {} ...'.format(self.parse_tree_path))
            if self.language == 'zh':
                try:
                    import pyltp
                    self.build_pos_to_file_with_pyltp(self.parse_tree_path)
                except:
                    self.build_pos_to_file_with_stanza(self.parse_tree_path)
            else:
                self.build_pos_to_file_with_stanza(self.parse_tree_path)
            self.read_pos_from_file(self.parse_tree_path)
        # if os.path.exists(self.parse_tree_path) and not self.rebuild:
        #     logger = getLogger()
        #     logger.info('read pos infomation from {} ...'.format(self.parse_tree_path))
        #     self.read_pos_from_file(self.parse_tree_path)
        # else:
        #     logger = getLogger()
        #     logger.info('build pos infomation to {} ...'.format(self.parse_tree_path))
        #     self.build_pos_to_file(self.parse_tree_path)
        #     self.read_pos_from_file(self.parse_tree_path)

    def _build_vocab(self):
        words_count_1 = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count_1[word] += 1
                except:
                    words_count_1[word] = 1
        self.in_idx2word_1 = [SpecialTokens.PAD_TOKEN, SpecialTokens.UNK_TOKEN]
        for key, value in words_count_1.items():
            if value > self.min_word_keep or "NUM" in key:
                self.in_idx2word_1.append(key)
        words_count_2 = {}
        for data in self.trainset:
            words_list = data["pos"]
            for word in words_list:
                try:
                    words_count_2[word] += 1
                except:
                    words_count_2[word] = 1
        self.in_idx2word_2 = [SpecialTokens.PAD_TOKEN,SpecialTokens.UNK_TOKEN]
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

    def _build_symbol(self):
        if self.share_vocab:
            self.out_idx2symbol_2 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        else:
            self.out_idx2symbol_2 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        self.num_start2 = len(self.out_idx2symbol_2)
        self.out_idx2symbol_2 += self.generate_list
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
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
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))
        # for data in self.trainset:
        #     words_list = data["postfix equation"]
        #     for word in words_list:
        #         if word in self.out_idx2symbol_2:
        #             continue
        #         elif word[0].isdigit():
        #             continue
        #         elif (word[0].isalpha() or word[0].isdigit()) is not True:
        #             self.out_idx2symbol_2.insert(self.num_start2, word)
        #             self.num_start2 += 1
        #             continue
        #         else:
        #             self.out_idx2symbol_2.append(word)
        self.out_idx2symbol_2 += [SpecialTokens.SOS_TOKEN]
        self.out_idx2symbol_2 += [SpecialTokens.UNK_TOKEN]

    def _build_symbol_for_tree(self):
        self.out_idx2symbol_1 = copy.deepcopy(self.operator_list)
        self.num_start1 = len(self.out_idx2symbol_1)
        self.out_idx2symbol_1 += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                self.out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
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
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        self.out_idx2symbol_1 += [SpecialTokens.UNK_TOKEN]
    
    def build_pos_to_file_with_stanza(self,path):
        nlp = stanza.Pipeline(self.language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
        new_datas=[]
        for data in self.trainset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                #pos.append(token['xpos'])
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id':data['id'],'pos':pos,'parse tree':parse_tree})
        for data in self.validset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id':data['id'],'pos':pos,'parse tree':parse_tree})
        for data in self.testset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id':data['id'],'pos':pos,'parse tree':parse_tree})
        write_json_data(new_datas,path)
    
    def build_pos_to_file_with_pyltp(self,path):
        from pyltp import Postagger,Parser
        pos_model_path = os.path.join(self.ltp_model_path, "pos.model")
        par_model_path = os.path.join(self.ltp_model_path, 'parser.model')
        postagger = Postagger()
        postagger.load(pos_model_path)
        parser = Parser()
        parser.load(par_model_path)
        
        new_datas=[]
        for data in self.trainset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head-1 for arc in arcs]
            new_datas.append({'id':data['id'],'pos':postags,'parse tree':parse_tree})
        for data in self.validset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head-1 for arc in arcs]
            new_datas.append({'id':data['id'],'pos':postags,'parse tree':parse_tree})
        for data in self.testset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head-1 for arc in arcs]
            new_datas.append({'id':data['id'],'pos':postags,'parse tree':parse_tree})
        write_json_data(new_datas,path)
    
    def read_pos_from_file(self,path):
        pos_datas=read_json_data(path)
        for data in self.trainset:
            for pos_data in pos_datas:
                if pos_data['id']!=data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break
        for data in self.validset:
            for pos_data in pos_datas:
                if pos_data['id']!=data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break
        for data in self.testset:
            for pos_data in pos_datas:
                if pos_data['id']!=data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break
    