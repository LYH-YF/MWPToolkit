from functools import partial
from logging import getLogger
import os

import torch
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from mwptoolkit.config.configuration import Config
from mwptoolkit.evaluate.evaluator import AbstractEvaluator, SeqEvaluator, PostEvaluator, PreEvaluator, MultiWayTreeEvaluator
from mwptoolkit.data.utils import create_dataset, create_dataloader
from mwptoolkit.utils.utils import get_model, init_seed, get_trainer
from mwptoolkit.utils.enum_type import SpecialTokens, FixType
from mwptoolkit.utils.logger import init_logger

def train_process(search_parameter,configs):
    for key,value in search_parameter.items():
        configs[key]=value
    dataset = create_dataset(configs)
    dataset.dataset_load()

    dataloader = create_dataloader(configs)(configs, dataset)

    model = get_model(configs["model"])(configs, dataset).to(configs["device"])

    if configs["equation_fix"] == FixType.Prefix:
        evaluator = PreEvaluator(configs["out_symbol2idx"], configs["out_idx2symbol"], configs)
    elif configs["equation_fix"] == FixType.Nonfix:
        evaluator = SeqEvaluator(configs["out_symbol2idx"], configs["out_idx2symbol"], configs)
    elif configs["equation_fix"] == FixType.Postfix:
        evaluator = PostEvaluator(configs["out_symbol2idx"], configs["out_idx2symbol"], configs)
    else:
        raise NotImplementedError

    trainer = get_trainer(configs["task_type"], configs["model"], configs["supervising_mode"])(configs, model, dataloader, evaluator)
    trainer.param_search()

def hyper_search_process(model_name, dataset_name, task_type, search_parameter, config_dict={}):
    configs = Config(model_name, dataset_name, task_type, config_dict)

    init_seed(configs['random_seed'], True)

    init_logger(configs)
    logger = getLogger()

    logger.info(configs)
    ray.init(num_gpus=configs['gpu_nums'])

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["accuracy"])
    result=tune.run(
        partial(train_process,configs=configs),
        resources_per_trial={"cpu": 2, "gpu": configs['gpu_nums']},
        config=search_parameter,
        scheduler=scheduler,
        num_samples=1,
        progress_reporter=reporter
    )

    print("Best config: ", result.get_best_config(metric="accuracy", mode="max"))