import copy
import argparse
from logging import getLogger

import torch

from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.utils.enum_type import Operators, SpecialTokens, NumMask
from mwptoolkit.data.dataloader.single_equation_dataloader import SingleEquationDataLoader
from mwptoolkit.model.Seq2Tree.gts import GTS
from mwptoolkit.model.Seq2Tree.tsn import TSN
from mwptoolkit.evaluate.evaluator import PreEvaluator
from mwptoolkit.trainer.supervised_trainer import GTSTrainer,TSNTrainer
from mwptoolkit.config.configuration import Config
from mwptoolkit.utils.logger import init_logger
from mwptoolkit.utils.utils import init_seed

class AsdivGTSDataset(TemplateDataset):
    def __init__(self, config):
        super().__init__(config)

    def _preprocess(self):
        for d in self.trainset:
            copy_num = len(d['number list'])
            if copy_num > self.copy_nums:
                self.copy_nums = copy_num
        self.operator_list = copy.deepcopy(Operators.Single)
        self.special_token_list = [SpecialTokens.PAD_TOKEN, SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.UNK_TOKEN]
        self.operator_nums=len(self.operator_list)
        for data in self.trainset:
            data['infix equation']=data['equation']
            data['template']=data['equation']
        for data in self.validset:
            data['infix equation']=data['equation']
            data['template']=data['equation']
        for data in self.testset:
            data['infix equation']=data['equation']
            data['template']=data['equation']

    def _build_symbol(self):
        self.out_idx2symbol = copy.deepcopy(Operators.Single)
        self.num_start = len(self.out_idx2symbol)
        self.out_idx2symbol += self.generate_list

        mask_list = NumMask.number
        self.out_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
        self.out_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def _build_template_symbol(self):
        self.temp_idx2symbol = [SpecialTokens.OPT_TOKEN]
        self.temp_num_start = len(self.temp_idx2symbol)
        self.temp_idx2symbol += self.generate_list

        mask_list = NumMask.number
        self.temp_idx2symbol += [mask_list[i] for i in range(self.copy_nums)]
        self.temp_idx2symbol += [SpecialTokens.UNK_TOKEN]

    def get_dataset_len(self):
        return len(self.trainset),len(self.validset),len(self.testset)

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
    dataset = AsdivGTSDataset(config)
    logger.info("start training with {} fold cross validation.".format(config["k_fold"]))
    for fold_t in dataset.cross_validation_load(config["k_fold"], start_fold_t):
        
        config["fold_t"] = fold_t
        config["best_folds_accuracy"] = best_folds_accuracy
        

        dataloader = SingleEquationDataLoader(config, dataset)
        if config['model'].lower() == 'gts':
            model = GTS(config,dataset).to(config["device"])
        elif config['model'].lower() == 'tsn':
            model = TSN(config,dataset).to(config["device"])
        

        evaluator = PreEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        
        if config['model'].lower() == 'gts':
            trainer = GTSTrainer(config, model, dataloader, evaluator)
        elif config['model'].lower() == 'tsn':
            trainer = TSNTrainer(config, model, dataloader, evaluator)

        logger.info("fold {}".format(fold_t))
        if config["test_only"]:
            trainer.test()
            return
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
        'dataset_path':'mawps',
        'read_local_folds':True
    }

    config=Config(args.model, args.dataset, args.task_type, config_dict)
    train_cross_validation(config)
