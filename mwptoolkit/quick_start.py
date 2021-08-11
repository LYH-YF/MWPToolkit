from logging import getLogger
import torch

from mwptoolkit.config.configuration import Config
from mwptoolkit.evaluate.evaluator import AbstractEvaluator, SeqEvaluator, PostEvaluator, PreEvaluator, MultiWayTreeEvaluator
from mwptoolkit.evaluate.evaluator import MultiEncDecEvaluator
from mwptoolkit.data.utils import create_dataset, create_dataloader
from mwptoolkit.utils.utils import get_model, init_seed, get_trainer
from mwptoolkit.utils.enum_type import SpecialTokens, FixType
from mwptoolkit.utils.logger import init_logger


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
            evaluator = PreEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        elif config["equation_fix"] == FixType.Nonfix:
            evaluator = SeqEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        elif config["equation_fix"] == FixType.Postfix:
            evaluator = PostEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        elif config["equation_fix"] == FixType.MultiWayTree:
            evaluator = MultiWayTreeEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        else:
            raise NotImplementedError
        
        if config['model'].lower() in ['multiencdec']:
            evaluator = MultiEncDecEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)


        trainer = get_trainer(config["task_type"], config["model"],config["supervising_mode"],config)(config, model, dataloader, evaluator)
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


def run_toolkit(model_name, dataset_name, task_type, config_dict={}):
    config = Config(model_name, dataset_name, task_type, config_dict)

    init_seed(config['random_seed'], True)

    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)

    if config["k_fold"] != None:
        train_cross_validation(config)
    else:
        dataset.dataset_load()
        
        dataloader = create_dataloader(config)(config, dataset)

        model = get_model(config["model"])(config,dataset).to(config["device"])
        # if config["pretrained_model_path"]:
        #     config["vocab_size"] = len(model.tokenizer)
        #     config["symbol_size"] = len(model.tokenizer)
        #     config["embedding_size"] = len(model.tokenizer)
        #     config["in_word2idx"] = model.tokenizer.get_vocab()
        #     config["in_idx2word"] = list(model.tokenizer.get_vocab().keys())
        #     config["out_symbol2idx"] = model.tokenizer.get_vocab()
        #     config["out_idx2symbol"] = list(model.tokenizer.get_vocab().keys())

        if config["equation_fix"] == FixType.Prefix:
            evaluator = PreEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        elif config["equation_fix"] == FixType.Nonfix:
            evaluator = SeqEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        elif config["equation_fix"] == FixType.Postfix:
            evaluator = PostEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)
        else:
            raise NotImplementedError
        
        if config['model'].lower() in ['multiencdec']:
            evaluator = MultiEncDecEvaluator(config["out_symbol2idx"], config["out_idx2symbol"], config)

        trainer = get_trainer(config["task_type"], config["model"], config["supervising_mode"],config)(config, model, dataloader, evaluator)
        logger.info(model)
        if config["test_only"]:
            trainer.test()
        else:
            trainer.fit()
