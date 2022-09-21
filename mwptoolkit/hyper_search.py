# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:36:19
# @File: hyper_search.py


import os
import sys
from functools import partial
from logging import getLogger

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler

from mwptoolkit.config.configuration import Config
from mwptoolkit.evaluate.evaluator import AbstractEvaluator, InfixEvaluator, PostfixEvaluator, PrefixEvaluator, MultiWayTreeEvaluator
from mwptoolkit.evaluate.evaluator import MultiEncDecEvaluator
from mwptoolkit.data.utils import create_dataset, create_dataloader
from mwptoolkit.utils.utils import get_model, init_seed, get_trainer, read_json_data, write_json_data
from mwptoolkit.utils.enum_type import SpecialTokens, FixType
from mwptoolkit.utils.logger import init_logger

from mwptoolkit.quick_start import run_toolkit

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

def train_process(search_parameter,configs):
    for key,value in search_parameter.items():
        configs[key]=value
    dataset = create_dataset(configs)
    dataset.dataset_load()

    dataloader = create_dataloader(configs)(configs, dataset)

    model = get_model(configs["model"])(configs, dataset).to(configs["device"])

    if configs["equation_fix"] == FixType.Prefix:
        evaluator = PrefixEvaluator(configs)
    elif configs["equation_fix"] == FixType.Nonfix or configs["equation_fix"] == FixType.Infix:
        evaluator = InfixEvaluator(configs)
    elif configs["equation_fix"] == FixType.Postfix:
        evaluator = PostfixEvaluator(configs)
    elif configs["equation_fix"] == FixType.MultiWayTree:
        evaluator = MultiWayTreeEvaluator(configs)
    else:
        raise NotImplementedError
    
    if configs['model'].lower() in ['multiencdec']:
        evaluator = MultiEncDecEvaluator(configs)

    trainer = get_trainer(configs)(configs, model, dataloader, evaluator)
    trainer.param_search()

def hyper_search_process(model_name, dataset_name, task_type, search_parameter, config_dict={}):
    configs = Config(model_name, dataset_name, task_type, config_dict)

    init_seed(configs['random_seed'], True)

    init_logger(configs)
    logger = getLogger()

    logger.info(configs)
    ray.init(num_gpus=configs['gpu_nums'])

    scheduler = AsyncHyperBandScheduler(
        metric="accuracy",
        mode="max")
    result=tune.run(
        partial(train_process,configs=configs),
        resources_per_trial={"cpu": configs['cpu_per_trial'], "gpu": configs['gpu_per_trial']},
        config=search_parameter,
        scheduler=scheduler,
        num_samples=configs["samples"],
        raise_on_failed_trial=False
    )
    best_config=result.get_best_config(metric="accuracy", mode="max")

    logger.info("best config:{}".format(best_config))
    
    config_dict.update(best_config)

    model_config_path = configs["model_config_file"]
    if not os.path.isabs(model_config_path):
        model_config_path = os.path.join(os.getcwd(),model_config_path)
    model_config=read_json_data(model_config_path)

    model_config.update(best_config)
    best_config_path = configs["best_config_file"]
    if not os.path.isabs(best_config_path):
        best_config_path = os.path.join(os.getcwd(),best_config_path)
    write_json_data(model_config,best_config_path)
    logger.info("best config saved at {}".format(best_config_path))

    run_toolkit(model_name,dataset_name,task_type,config_dict)
