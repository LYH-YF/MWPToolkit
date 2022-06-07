# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:33:11
# @File: dataset_multiencdec.py
import json
import os
import copy
from logging import getLogger
import stanza

from mwptoolkit.config.configuration import Config
from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens, FixType, Operators, MaskSymbol, SPECIAL_TOKENS, \
    DatasetName, TaskType
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, \
    from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tools import id_reedit, dataset_drop_duplication
from mwptoolkit.utils.preprocess_tool.number_transfer import number_transfer
from mwptoolkit.utils.utils import read_json_data, write_json_data


class DatasetMultiEncDec(TemplateDataset):
    """dataset class for deep-learning model MultiE&D
    """

    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

        task_type (str): [single_equation | multi_equation], the type of task.

        parse_tree_file_name (str|None): the name of the file to save parse tree information.

        ltp_model_dir or ltp_model_path (str|None): the road path of ltp model.

        model (str): model name.

        dataset (str): dataset name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.
        
        dataset_dir or dataset_path (str): the road path of dataset folder.

        language (str): a property of dataset, the language of dataset.

        single (bool): a property of dataset, the equation of dataset is single or not.

        linear (bool): a property of dataset, the equation of dataset is linear or not.

        source_equation_fix (str): [infix | postfix | prefix], a property of dataset, the source format of equation of dataset.

        rebuild (bool): when loading additional dataset information, this can decide to build information anew or load information built before.

        validset_divide (bool): whether to split validset. if True, the dataset is split to trainset-validset-testset. if False, the dataset is split to trainset-testset.

        mask_symbol (str): [NUM | number], the symbol to mask numbers in equation.

        min_word_keep (int): in dataset, words that count greater than the value, will be kept in input vocabulary.

        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        k_fold (int|None): if it's an integer, it indicates to run k-fold cross validation. if it's None, it indicates to run trainset-validset-testset split.

        read_local_folds (bool): when running k-fold cross validation, if True, then loading split folds from dataset folder. if False, randomly split folds.

        shuffle (bool): whether to shuffle trainset before training.

        device (torch.device):

        resume_training or resume (bool):
        """
        super().__init__(config)
        self.task_type = config['task_type']
        self.parse_tree_path = config['parse_tree_file_name']
        if self.parse_tree_path is not None:
            self.parse_tree_path = os.path.join(self.dataset_path, self.parse_tree_path + '.json')
            if not os.path.isabs(self.parse_tree_path):
                self.parse_tree_path = os.path.join(os.getcwd(), self.parse_tree_path)

        self.ltp_model_path = config['ltp_model_dir'] if config['ltp_model_dir'] else config['ltp_model_path']
        if self.ltp_model_path and not os.path.isabs(self.ltp_model_path):
            self.ltp_model_path = os.path.join(os.getcwd(), self.ltp_model_path)

    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset, self.validset, self.testset = id_reedit(self.trainset, self.validset, self.testset)
        if self.dataset in [DatasetName.draw]:
            self.trainset, self.validset, self.testset = dataset_drop_duplication(self.trainset, self.validset,
                                                                                  self.testset)
        transfer = number_transfer

        self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(self.trainset, self.dataset,
                                                                             self.task_type, self.mask_symbol,
                                                                             self.min_generate_keep,self.linear, ";")
        self.validset, _g, valid_copy_nums, _ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol,
                                                         self.min_generate_keep,self.linear, ";")
        self.testset, _g, test_copy_nums, _ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol,
                                                       self.min_generate_keep,self.linear, ";")
        source_equation_fix = self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if source_equation_fix == FixType.Infix:
            to_infix = None
            to_prefix = from_infix_to_prefix
            to_postfix = from_infix_to_postfix
        elif source_equation_fix == FixType.Prefix:
            to_infix = from_prefix_to_infix
            to_prefix = None
            to_postfix = from_prefix_to_postfix
        elif source_equation_fix == FixType.Postfix:
            to_infix = from_postfix_to_infix
            to_prefix = from_postfix_to_prefix
            to_postfix = None
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

        generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
        else:
            copy_nums = train_copy_nums

        if self.task_type == TaskType.SingleEquation:
            operator_list = copy.deepcopy(Operators.Single)
            if self.dataset in [DatasetName.mawps]:
                operator_list.append('=')
            operator_nums = len(operator_list)
        elif self.task_type == TaskType.MultiEquation:
            operator_nums = len(Operators.Multi)
            operator_list = copy.deepcopy(Operators.Multi)
        else:
            raise NotImplementedError
        if os.path.exists(self.parse_tree_path) and not self.rebuild:
            logger = getLogger()
            logger.info('read pos information from {} ...'.format(self.parse_tree_path))
            self.read_pos_from_file(self.parse_tree_path)
        else:
            logger = getLogger()
            logger.info('build pos information to {} ...'.format(self.parse_tree_path))
            if self.language == 'zh':
                try:
                    import pyltp
                    self.build_pos_to_file_with_pyltp(self.parse_tree_path)
                except:
                    self.build_pos_to_file_with_stanza(self.parse_tree_path)
            else:
                self.build_pos_to_file_with_stanza(self.parse_tree_path)
            self.read_pos_from_file(self.parse_tree_path)

        return {'generate_list': generate_list, 'copy_nums': copy_nums, 'operator_list': operator_list,
                'operator_nums': operator_nums}

    def _build_vocab(self):
        words_count_1 = {}
        for data in self.trainset:
            words_list = data["question"]
            for word in words_list:
                try:
                    words_count_1[word] += 1
                except:
                    words_count_1[word] = 1
        in_idx2word_1 = [SpecialTokens.PAD_TOKEN, SpecialTokens.UNK_TOKEN]
        for key, value in words_count_1.items():
            if value > self.min_word_keep or "NUM" in key:
                in_idx2word_1.append(key)
        words_count_2 = {}
        for data in self.trainset:
            words_list = data["pos"]
            for word in words_list:
                try:
                    words_count_2[word] += 1
                except:
                    words_count_2[word] = 1
        in_idx2word_2 = [SpecialTokens.PAD_TOKEN, SpecialTokens.UNK_TOKEN]
        for key, value in words_count_2.items():
            if value > self.min_word_keep:
                in_idx2word_2.append(key)

        equ_dict_2 = self._build_symbol()
        equ_dict_1 = self._build_symbol_for_tree()
        out_idx2symbol_2 = equ_dict_2['out_idx2symbol_2']
        out_idx2symbol_1 = equ_dict_1['out_idx2symbol_1']
        num_start1 = equ_dict_1['num_start1']
        num_start2 = equ_dict_2['num_start2']
        in_word2idx_1 = {}
        in_word2idx_2 = {}
        out_symbol2idx_1 = {}
        out_symbol2idx_2 = {}
        for idx, word in enumerate(in_idx2word_1):
            in_word2idx_1[word] = idx
        for idx, word in enumerate(in_idx2word_2):
            in_word2idx_2[word] = idx
        for idx, symbol in enumerate(out_idx2symbol_1):
            out_symbol2idx_1[symbol] = idx
        for idx, symbol in enumerate(out_idx2symbol_2):
            out_symbol2idx_2[symbol] = idx

        return {'in_idx2word_1': in_idx2word_1, 'in_idx2word_2': in_idx2word_2,
                'in_word2idx_1': in_word2idx_1, 'in_word2idx_2': in_word2idx_2,
                'out_idx2symbol_1': out_idx2symbol_1, 'out_symbol2idx_1': out_symbol2idx_1,
                'out_idx2symbol_2': out_idx2symbol_2, 'out_symbol2idx_2': out_symbol2idx_2,
                'num_start1': num_start1, 'num_start2': num_start2,
                }

    def _build_symbol(self):
        if self.share_vocab:
            out_idx2symbol_2 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        else:
            out_idx2symbol_2 = [SpecialTokens.PAD_TOKEN] + [SpecialTokens.EOS_TOKEN] + self.operator_list
        num_start2 = len(out_idx2symbol_2)
        out_idx2symbol_2 += self.generate_list
        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(
                        self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                out_idx2symbol_2 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError(
                    "{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.generate_list))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        out_idx2symbol_2 += [SpecialTokens.SOS_TOKEN]
        out_idx2symbol_2 += [SpecialTokens.UNK_TOKEN]
        return {'out_idx2symbol_2': out_idx2symbol_2, 'num_start2': num_start2}

    def _build_symbol_for_tree(self):
        out_idx2symbol_1 = copy.deepcopy(self.operator_list)
        num_start1 = len(out_idx2symbol_1)
        out_idx2symbol_1 += self.generate_list

        if self.mask_symbol == MaskSymbol.NUM:
            mask_list = NumMask.number
            try:
                out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        elif self.mask_symbol == MaskSymbol.alphabet:
            mask_list = NumMask.alphabet
            try:
                out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError(
                    "alphabet may not enough to mask {} numbers, changing the mask_symbol from alphabet to number may solve the problem.".format(
                        self.copy_nums))
        elif self.mask_symbol == MaskSymbol.number:
            mask_list = NumMask.number
            try:
                out_idx2symbol_1 += [mask_list[i] for i in range(self.copy_nums)]
            except IndexError:
                raise IndexError("{} numbers is not enough to mask {} numbers ".format(len(mask_list), self.copy_nums))
        else:
            raise NotImplementedError("the type of masking number ({}) is not implemented".format(self.mask_symbol))

        out_idx2symbol_1 += [SpecialTokens.UNK_TOKEN]
        return {'out_idx2symbol_1': out_idx2symbol_1, 'num_start1': num_start1}

    def build_pos_to_file_with_stanza(self, path):
        nlp = stanza.Pipeline(self.language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True,
                              logging_level='error')
        new_datas = []
        for data in self.trainset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                # pos.append(token['xpos'])
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id': data['id'], 'pos': pos, 'parse tree': parse_tree})
        for data in self.validset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id': data['id'], 'pos': pos, 'parse tree': parse_tree})
        for data in self.testset:
            doc = nlp(data["ques source 1"])
            token_list = doc.to_dict()[0]
            pos = []
            parse_tree = []
            for token in token_list:
                pos.append(token['upos'])
                parse_tree.append(token['head'] - 1)
            new_datas.append({'id': data['id'], 'pos': pos, 'parse tree': parse_tree})
        write_json_data(new_datas, path)

    def build_pos_to_file_with_pyltp(self, path):
        from pyltp import Postagger, Parser
        pos_model_path = os.path.join(self.ltp_model_path, "pos.model")
        par_model_path = os.path.join(self.ltp_model_path, 'parser.model')
        postagger = Postagger()
        postagger.load(pos_model_path)
        parser = Parser()
        parser.load(par_model_path)

        new_datas = []
        for data in self.trainset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head - 1 for arc in arcs]
            new_datas.append({'id': data['id'], 'pos': postags, 'parse tree': parse_tree})
        for data in self.validset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head - 1 for arc in arcs]
            new_datas.append({'id': data['id'], 'pos': postags, 'parse tree': parse_tree})
        for data in self.testset:
            postags = postagger.postag(data["ques source 1"].split(' '))
            postags = ' '.join(postags).split(' ')
            arcs = parser.parse(data["ques source 1"].split(' '), postags)
            parse_tree = [arc.head - 1 for arc in arcs]
            new_datas.append({'id': data['id'], 'pos': postags, 'parse tree': parse_tree})
        write_json_data(new_datas, path)

    def read_pos_from_file(self, path):
        pos_datas = read_json_data(path)
        for data in self.trainset:
            for pos_data in pos_datas:
                if pos_data['id'] != data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break
        for data in self.validset:
            for pos_data in pos_datas:
                if pos_data['id'] != data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break
        for data in self.testset:
            for pos_data in pos_datas:
                if pos_data['id'] != data['id']:
                    continue
                else:
                    data['pos'] = pos_data['pos']
                    data['parse tree'] = pos_data['parse tree']
                    pos_datas.remove(pos_data)
                    break

    def save_dataset(self, save_dir: str):
        """
        save dataset parameters to file.

        :param save_dir: (str) folder which saves the parameter file
        :return:
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        input_vocab_file = os.path.join(save_dir, 'input_vocab.json')
        write_json_data(
            {
                'in_idx2word_1': self.in_idx2word_1,
                'in_idx2word_2': self.in_idx2word_2
            },
            input_vocab_file
        )
        output_vocab_file = os.path.join(save_dir, 'output_vocab.json')
        write_json_data(
            {
                'out_idx2symbol_1': self.out_idx2symbol_1,
                'out_idx2symbol_2': self.out_idx2symbol_2
            },
            output_vocab_file
        )
        data_id_file = os.path.join(save_dir, 'data_split.json')
        write_json_data(
            {
                'trainset_id': self.trainset_id,
                'validset_id': self.validset_id,
                'testset_id': self.testset_id,
                'folds_id': self.folds_id
            },
            data_id_file
        )
        json_encoder = json.encoder.JSONEncoder()
        parameters_dict = self.parameters_to_dict()
        not_support_json = []
        not_to_save = ['in_idx2word_1', 'in_idx2word_2', 'out_idx2symbol_1', 'out_idx2symbol_2',
                       'in_word2idx_1', 'in_word2idx_2', 'out_symbol2idx_1', 'out_symbol2idx_2',
                       'folds', 'trainset', 'testset', 'validset', 'datas', 'trainset_id',
                       'validset_id', 'testset_id', 'folds_id']
        for key, value in parameters_dict.items():
            try:
                json_encoder.encode({key: value})
            except TypeError:
                not_support_json.append(key)
        for key in not_support_json:
            del parameters_dict[key]
        for key in not_to_save:
            del parameters_dict[key]
        parameter_file = os.path.join(save_dir, 'dataset.json')
        write_json_data(parameters_dict, parameter_file)

    @classmethod
    def load_from_pretrained(cls, pretrained_dir: str, resume_training=False):
        """
        load dataset parameters from file.

        :param pretrained_dir: (str) folder which saved the parameter file
        :param resume_training: (bool) load parameter for resuming training or not.
        :return: an instantiated object
        """
        config = Config.load_from_pretrained(pretrained_dir)
        dataset = DatasetMultiEncDec(config)

        input_vocab_file = os.path.join(pretrained_dir, 'input_vocab.json')
        output_vocab_file = os.path.join(pretrained_dir, 'output_vocab.json')
        parameter_file = os.path.join(pretrained_dir, 'dataset.json')
        data_id_file = os.path.join(pretrained_dir, 'data_split.json')

        input_vocab = read_json_data(input_vocab_file)
        output_vocab = read_json_data(output_vocab_file)
        parameter_dict = read_json_data(parameter_file)
        data_id_dict = read_json_data(data_id_file)

        in_idx2word_1 = input_vocab['in_idx2word_1']
        in_idx2word_2 = input_vocab['in_idx2word_2']
        out_idx2symbol_1 = output_vocab['out_idx2symbol_1']
        out_idx2symbol_2 = output_vocab['out_idx2symbol_2']

        in_word2idx_1 = {}
        in_word2idx_2 = {}
        out_symbol2idx_1 = {}
        out_symbol2idx_2 = {}
        for idx, word in enumerate(in_idx2word_1):
            in_idx2word_1[word] = idx
        for idx, word in enumerate(in_idx2word_2):
            in_idx2word_2[word] = idx
        for idx, symbol in enumerate(out_idx2symbol_1):
            out_idx2symbol_1[symbol] = idx
        for idx, symbol in enumerate(out_idx2symbol_2):
            out_idx2symbol_2[symbol] = idx

        setattr(dataset, 'in_idx2word_1', in_idx2word_1)
        setattr(dataset, 'in_idx2word_2', in_idx2word_2)
        setattr(dataset, 'out_idx2symbol_1', out_idx2symbol_1)
        setattr(dataset, 'out_idx2symbol_2', out_idx2symbol_2)
        setattr(dataset, 'in_word2idx_1', in_word2idx_1)
        setattr(dataset, 'in_word2idx_2', in_word2idx_2)
        setattr(dataset, 'out_symbol2idx_1', out_symbol2idx_1)
        setattr(dataset, 'out_symbol2idx_2', out_symbol2idx_2)
        for key, value in parameter_dict.items():
            setattr(dataset, key, value)
        for key,value in data_id_dict.items():
            setattr(dataset, key, value)
        if resume_training:
            if config['k_fold']:
                setattr(dataset, 'fold_t', config['fold_t'])
                setattr(dataset, 'the_fold_t', config['fold_t'] - 1)
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
            else:
                setattr(dataset, 'from_pretrained', False)
                setattr(dataset, 'pretrained_dir', pretrained_dir)
                setattr(dataset, 'resume_training', resume_training)
        else:
            setattr(dataset, 'from_pretrained', True)
            setattr(dataset, 'pretrained_dir', pretrained_dir)
        dataset.reset_dataset()
        return dataset

    def __load_pretrained_parameters(self):
        if self.k_fold:
            load_dir = os.path.join(self.pretrained_dir, 'fold{}'.format(self.fold_t))
        else:
            load_dir = self.pretrained_dir

        input_vocab_file = os.path.join(load_dir, 'input_vocab.json')
        output_vocab_file = os.path.join(load_dir, 'output_vocab.json')
        parameter_file = os.path.join(load_dir, 'dataset.json')

        input_vocab = read_json_data(input_vocab_file)
        output_vocab = read_json_data(output_vocab_file)
        parameter_dict = read_json_data(parameter_file)

        in_idx2word_1 = input_vocab['in_idx2word_1']
        in_idx2word_2 = input_vocab['in_idx2word_2']
        out_idx2symbol_1 = output_vocab['out_idx2symbol_1']
        out_idx2symbol_2 = output_vocab['out_idx2symbol_2']

        in_word2idx_1 = {}
        in_word2idx_2 = {}
        out_symbol2idx_1 = {}
        out_symbol2idx_2 = {}
        for idx, word in enumerate(in_idx2word_1):
            in_idx2word_1[word] = idx
        for idx, word in enumerate(in_idx2word_2):
            in_idx2word_2[word] = idx
        for idx, symbol in enumerate(out_idx2symbol_1):
            out_idx2symbol_1[symbol] = idx
        for idx, symbol in enumerate(out_idx2symbol_2):
            out_idx2symbol_2[symbol] = idx

        setattr(self, 'in_idx2word_1', in_idx2word_1)
        setattr(self, 'in_idx2word_2', in_idx2word_2)
        setattr(self, 'out_idx2symbol_1', out_idx2symbol_1)
        setattr(self, 'out_idx2symbol_2', out_idx2symbol_2)
        setattr(self, 'in_word2idx_1', in_word2idx_1)
        setattr(self, 'in_word2idx_2', in_word2idx_2)
        setattr(self, 'out_symbol2idx_1', out_symbol2idx_1)
        setattr(self, 'out_symbol2idx_2', out_symbol2idx_2)
        for key, value in parameter_dict.items():
            setattr(self, key, value)
