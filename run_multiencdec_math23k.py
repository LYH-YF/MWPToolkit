import copy
import argparse
from logging import getLogger

import torch

from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.enum_type import Operators, SpecialTokens, NumMask
from mwptoolkit.data.dataloader.dataloader_multiencdec import DataLoaderMultiEncDec
from mwptoolkit.model.Graph2Tree.multiencdec import MultiEncDec
from mwptoolkit.evaluate.evaluator import MultiEncDecEvaluator
from mwptoolkit.trainer.supervised_trainer import GTSTrainer,TSNTrainer,MultiEncDecTrainer
from mwptoolkit.config.configuration import Config
from mwptoolkit.utils.logger import init_logger
from mwptoolkit.utils.utils import init_seed
from mwptoolkit.utils.enum_type import MaskSymbol

class Math23kMultiEncDecDataset(TemplateDataset):
    def __init__(self, config):
        super().__init__(config)

    def _preprocess(self):
        for d in self.trainset:
            copy_num = len(d['number list'])
            if copy_num > self.copy_nums:
                self.copy_nums = copy_num
        for d in self.validset:
            copy_num = len(d['number list'])
            if copy_num > self.copy_nums:
                self.copy_nums = copy_num
        for d in self.testset:
            copy_num = len(d['number list'])
            if copy_num > self.copy_nums:
                self.copy_nums = copy_num
        self.operator_list = copy.deepcopy(Operators.Single)
        self.special_token_list = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.UNK_TOKEN]
        self.operator_nums=len(self.operator_list)
        for data in self.trainset:
            data['ans']='-'
        for data in self.validset:
            data['ans']='-'
        for data in self.testset:
            data['ans']='-'
        generate_num_count={}
        for data in self.trainset:
            for symbol in data['prefix equation']:
                if symbol[0].isdigit() and symbol not in data['number list']:
                    if symbol not in generate_num_count:
                        generate_num_count[symbol]=1
                    else:
                        generate_num_count[symbol]+=1
        for symbol,counts in generate_num_count.items():
            if counts>self.min_generate_keep:
                self.generate_list.append(symbol)
        for data in self.validset:
            data['ans']='-'
        for data in self.testset:
            data['ans']='-'
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
    

def train_cross_validation(config):
    init_logger(config)
    init_seed(config['random_seed'], True)

    if config["resume"]:
        check_pnt = torch.load(config["checkpoint_path"], map_location=config["map_location"])
        start_fold_t = check_pnt["fold_t"]
        best_folds_accuracy = check_pnt["best_folds_accuracy"]
    else:
        start_fold_t = 0
        best_folds_accuracy = []
    logger = getLogger()
    logger.info(config)
    dataset = Math23kMultiEncDecDataset(config)
    logger.info("start training with {} fold cross validation.".format(config["k_fold"]))
    for fold_t in dataset.cross_validation_load(config["k_fold"], start_fold_t):
        
        config["fold_t"] = fold_t
        config["best_folds_accuracy"] = best_folds_accuracy
        

        dataloader = DataLoaderMultiEncDec(config, dataset)
        model = MultiEncDec(config,dataset)

        evaluator = MultiEncDecEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        
        
        trainer = MultiEncDecTrainer(config, model, dataloader, evaluator)

        logger.info("fold {}".format(fold_t))
        if config["test_only"]:
            trainer.test()
            best_folds_accuracy.append({"fold_t": fold_t, "best_equ_accuracy": trainer.best_test_equ_accuracy, "best_value_accuracy": trainer.best_test_value_accuracy})
        else:
            trainer.fit()
            best_folds_accuracy.append({"fold_t": fold_t, "best_equ_accuracy": trainer.best_test_equ_accuracy, "best_value_accuracy": trainer.best_test_value_accuracy})
    best_folds_accuracy = sorted(best_folds_accuracy, key=lambda x: x["best_value_accuracy"], reverse=True)
    logger.info("{} fold cross validation finished.".format(config["k_fold"]))
    best_equ_accuracy = []
    best_value_accuracy = []
    for accuracy in best_folds_accuracy:
        best_equ_accuracy.append(accuracy["best_equ_accuracy"])
        best_value_accuracy.append(accuracy["best_value_accuracy"])
        logger.info("fold %2d : test equ accuracy [%2.3f] | test value accuracy [%2.3f]"\
                        %(accuracy["fold_t"],accuracy["best_equ_accuracy"],accuracy["best_value_accuracy"]))
    logger.info("folds avr : test equ accuracy [%2.3f] | test value accuracy [%2.3f]"\
                    %(sum(best_equ_accuracy)/len(best_equ_accuracy),sum(best_value_accuracy)/len(best_value_accuracy)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GTS', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='math23k', help='name of datasets')
    parser.add_argument('--task_type', '-t', type=str, default='single_equation', help='name of tasks')
    #parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    config_dict = {
        'dataset_path':'math23k',
        'read_local_folds':True
    }

    config=Config(args.model, args.dataset, args.task_type, config_dict)
    train_cross_validation(config)