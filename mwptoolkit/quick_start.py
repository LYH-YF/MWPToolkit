# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:31:08
# @File: quick_start.py


import os
import sys
from logging import getLogger

import torch

from mwptoolkit.config.configuration import Config
from mwptoolkit.evaluate.evaluator import AbstractEvaluator, InfixEvaluator, PostfixEvaluator, PrefixEvaluator, MultiWayTreeEvaluator
from mwptoolkit.evaluate.evaluator import MultiEncDecEvaluator,get_evaluator,get_evaluator_module
from mwptoolkit.data.utils import create_dataset, create_dataloader,get_dataset_module,get_dataloader_module
from mwptoolkit.utils.utils import get_model, init_seed, get_trainer
from mwptoolkit.utils.enum_type import SpecialTokens, FixType
from mwptoolkit.utils.logger import init_logger

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))


def train_cross_validation(config):
    if config["resume"]:
        check_pnt = torch.load(config["checkpoint_path"], map_location=config["map_location"])
        start_fold_t = check_pnt["fold_t"]
        best_folds_accuracy = check_pnt["best_folds_accuracy"]
    else:
        start_fold_t = 0
        best_folds_accuracy = []
    logger = getLogger()
    dataset = create_dataset(config)
    logger.info("start training with {} fold cross validation.".format(config["k_fold"]))
    for fold_t in dataset.cross_validation_load(config["k_fold"], start_fold_t):
        config["fold_t"] = fold_t
        config["best_folds_accuracy"] = best_folds_accuracy

        dataloader = create_dataloader(config)(config, dataset)

        model = get_model(config["model"])(config,dataset).to(config["device"])

        if config["equation_fix"] == FixType.Prefix:
            evaluator = PrefixEvaluator(config)
        elif config["equation_fix"] == FixType.Nonfix or config["equation_fix"] == FixType.Infix:
            evaluator = InfixEvaluator(config)
        elif config["equation_fix"] == FixType.Postfix:
            evaluator = PostfixEvaluator(config)
        elif config["equation_fix"] == FixType.MultiWayTree:
            evaluator = MultiWayTreeEvaluator(config)
        else:
            raise NotImplementedError
        
        if config['model'].lower() in ['multiencdec']:
            evaluator = MultiEncDecEvaluator(config)


        trainer = get_trainer(config)(config, model, dataloader, evaluator)
        logger.info("fold {}".format(fold_t))
        if config["test_only"]:
            trainer.test()
            best_folds_accuracy.append({"fold_t": fold_t, "best_equ_accuracy": trainer.best_test_equ_accuracy, "best_value_accuracy": trainer.best_test_value_accuracy})
        else:
            trainer.fit()
            best_folds_accuracy.append({"fold_t": fold_t, "best_equ_accuracy": trainer.best_test_equ_accuracy, "best_value_accuracy": trainer.best_test_value_accuracy})
        config["resume"]=False
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


def train_with_train_valid_test_split(temp_config):
    if temp_config['training_resume'] or temp_config['resume']:
        config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
    else:
        config = temp_config

    logger = getLogger()
    logger.info(config)
    if temp_config['training_resume'] or temp_config['resume']:
        dataset = get_dataset_module(config).load_from_pretrained(temp_config['checkpoint_dir'])
    else:
        dataset = get_dataset_module(config)(config)

    dataset.dataset_load()

    dataloader = get_dataloader_module(config)(config,dataset)

    model = get_model(config["model"])(config, dataset).to(config["device"])

    evaluator = get_evaluator_module(config)(config)

    trainer = get_trainer(config)(config, model, dataloader, evaluator)
    logger.info(model)
    trainer.fit()


def test_with_train_valid_test_split(temp_config):
    config = Config.load_from_pretrained(temp_config['trained_model_dir'])

    logger = getLogger()
    logger.info(config)

    dataset = get_dataset_module(config).load_from_pretrained(temp_config['trained_model_dir'])
    dataset.dataset_load()

    dataloader = get_dataloader_module(config)(config, dataset)

    model = get_model(config["model"])(config, dataset).to(config["device"])

    evaluator = get_evaluator_module(config)(config)

    trainer = get_trainer(config)(config, model, dataloader, evaluator)
    trainer.test()


def train_with_cross_validation(temp_config):
    if temp_config['training_resume'] or temp_config['resume']:
        config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
        fold_load_file = os.path.join(temp_config['checkpoint_dir'],'fold{}'.format(config['fold_t']))
        accuracy_load_file = os.path.join(fold_load_file,'trainer_checkpoint.pth')
        check_pnt = torch.load(accuracy_load_file, map_location='cpu')
        best_folds_accuracy = check_pnt["best_folds_accuracy"]
        config['training_resume'] = True
        config['resume'] = True
    else:
        config = temp_config
        best_folds_accuracy = []

    logger = getLogger()
    logger.info(config)

    if temp_config['training_resume'] or temp_config['resume']:
        dataset_file = os.path.join(temp_config['checkpoint_dir'],'fold{}'.format(config['fold_t']))
        dataset = get_dataset_module(config).load_from_pretrained(dataset_file,resume_training=True)
    else:
        dataset = get_dataset_module(config)(config)

    start_fold_t = dataset.fold_t
    for fold_t in range(start_fold_t,config['k_fold']):
        config['fold_t'] = fold_t
        config["best_folds_accuracy"] = best_folds_accuracy

        dataset.dataset_load()

        dataloader = get_dataloader_module(config)(config, dataset)

        model = get_model(config["model"])(config, dataset).to(config["device"])

        evaluator = get_evaluator_module(config)(config)

        trainer = get_trainer(config)(config, model, dataloader, evaluator)

        trainer.fit()
        best_folds_accuracy.append({"fold_t": fold_t, "best_equ_accuracy": trainer.best_test_equ_accuracy,
                                    "best_value_accuracy": trainer.best_test_value_accuracy})
        config["resume"] = False
        config['training_resume'] = False
    best_folds_accuracy = sorted(best_folds_accuracy, key=lambda x: x["best_value_accuracy"], reverse=True)
    logger.info("{} fold cross validation finished.".format(config["k_fold"]))
    best_equ_accuracy = []
    best_value_accuracy = []
    for accuracy in best_folds_accuracy:
        best_equ_accuracy.append(accuracy["best_equ_accuracy"])
        best_value_accuracy.append(accuracy["best_value_accuracy"])
        logger.info("fold %2d : test equ accuracy [%2.3f] | test value accuracy [%2.3f]" \
                    % (accuracy["fold_t"], accuracy["best_equ_accuracy"], accuracy["best_value_accuracy"]))
    logger.info("folds avr : test equ accuracy [%2.3f] | test value accuracy [%2.3f]" \
                % (sum(best_equ_accuracy) / len(best_equ_accuracy),
                   sum(best_value_accuracy) / len(best_value_accuracy)))


def test_with_cross_validation(temp_config):
    config = Config.load_from_pretrained(temp_config['trained_model_dir'])

    logger = getLogger()
    logger.info(config)
    dataset = get_dataset_module(config).load_from_pretrained(temp_config['trained_model_dir'])
    for fold_t in range(config['k_fold']):
        dataset.dataset_load()

        dataloader = get_dataloader_module(config)(config, dataset)

        model = get_model(config["model"])(config, dataset).to(config["device"])

        evaluator = get_evaluator_module(config)(config)

        trainer = get_trainer(config)(config, model, dataloader, evaluator)
        trainer.test()


def run_toolkit(model_name, dataset_name, task_type, config_dict={}):
    config = Config(model_name, dataset_name, task_type, config_dict)

    init_seed(config['random_seed'], True)

    init_logger(config)
    if config['test_only']:
        if config['k_fold'] is not None:
            test_with_cross_validation(config)
        else:
            test_with_train_valid_test_split(config)
    else:
        if config["k_fold"] is not None:
            train_with_cross_validation(config)
        else:
            train_with_train_valid_test_split(config)
