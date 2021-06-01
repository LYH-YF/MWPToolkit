from functools import partial
from logging import getLogger

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

def train_process(search_parameter,config):
    for key,value in search_parameter.items():
        config[key]=value
    dataset = create_dataset(config)
    dataset.dataset_load()
    if config["share_vocab"]:
        config["out_symbol2idx"] = dataset.out_symbol2idx
        config["out_idx2symbol"] = dataset.out_idx2symbol
        config["in_word2idx"] = dataset.in_word2idx
        config["in_idx2word"] = dataset.in_idx2word
        config["out_sos_token"] = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
    else:
        if config["symbol_for_tree"]:
            config["out_symbol2idx"] = dataset.out_symbol2idx
            config["out_idx2symbol"] = dataset.out_idx2symbol
            config["in_word2idx"] = dataset.in_word2idx
            config["in_idx2word"] = dataset.in_idx2word
        else:
            config["out_symbol2idx"] = dataset.out_symbol2idx
            config["out_idx2symbol"] = dataset.out_idx2symbol
            config["out_sos_token"] = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
            config["out_eos_token"] = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
            config["out_pad_token"] = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
            config["in_word2idx"] = dataset.in_word2idx
            config["in_idx2word"] = dataset.in_idx2word

    config["vocab_size"] = len(dataset.in_idx2word)
    config["symbol_size"] = len(dataset.out_idx2symbol)
    config['span_size'] = dataset.max_span_size
    config["operator_nums"] = dataset.operator_nums
    config["copy_nums"] = dataset.copy_nums
    config["generate_size"] = len(dataset.generate_list)
    config["generate_list"] = dataset.generate_list
    config["operator_list"] = dataset.operator_list
    config["num_start"] = dataset.num_start

    dataloader = create_dataloader(config)(config, dataset)

    model = get_model(config["model"])(config).to(config["device"])
    if config["pretrained_model_path"]:
        config["vocab_size"] = len(model.tokenizer)
        config["symbol_size"] = len(model.tokenizer)
        config["embedding_size"] = len(model.tokenizer)
        config["in_word2idx"] = model.tokenizer.get_vocab()
        config["in_idx2word"] = list(model.tokenizer.get_vocab().keys())
        config["out_symbol2idx"] = model.tokenizer.get_vocab()
        config["out_idx2symbol"] = list(model.tokenizer.get_vocab().keys())

    if config["equation_fix"] == FixType.Prefix:
        evaluator = PreEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
    elif config["equation_fix"] == FixType.Nonfix:
        evaluator = SeqEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
    elif config["equation_fix"] == FixType.Postfix:
        evaluator = PostEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
    else:
        raise NotImplementedError

    trainer = get_trainer(config["task_type"], config["model"])(config, model, dataloader, evaluator)
    trainer.fit()
    tune.report(accuracy=trainer.best_test_value_accuracy)

def hyper_search_process(model_name, dataset_name, task_type, search_parameter, config_dict={}):
    config = Config(model_name, dataset_name, task_type, config_dict)

    init_seed(config['random_seed'], True)

    init_logger(config)
    logger = getLogger()

    logger.info(config)
    ray.init(num_gpus=config['gpu_nums'])

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["accuracy"])
    result=tune.run(
        partial(train_process,config=config),
        config=search_parameter,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    print("Best config: ", result.get_best_config(metric="accuracy", mode="max"))