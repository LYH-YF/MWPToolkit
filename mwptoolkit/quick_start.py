from logging import getLogger
import torch

from mwptoolkit.config.configuration import Config
from mwptoolkit.evaluate.evaluator import SeqEvaluater, PostEvaluater, PreEvaluater
from mwptoolkit.data.utils import create_dataset, create_dataloader
from mwptoolkit.utils.utils import get_model, init_seed, get_trainer
from mwptoolkit.utils.enum_type import SpecialTokens, FixType
from mwptoolkit.utils.logger import init_logger


def train_cross_validation(config):
    if config["resume"]:
        check_pnt = torch.load(config["checkpoint_path"], map_location=config["map_location"])
        start_fold_t = check_pnt["fold_t"]
    else:
        start_fold_t = 0
    dataset = create_dataset(config)
    for fold_t in dataset.cross_validation_load(config["k_fold"], start_fold_t):
        if config["share_vocab"]:
            config["out_symbol2idx"] = dataset.out_symbol2idx
            config["out_idx2symbol"] = dataset.out_idx2symbol
            config["in_word2idx"] = dataset.in_word2idx
            config["in_idx2word"] = dataset.in_idx2word
            config["out_sos_token"] = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            if config["symbol_for_tree"]:
                pass
            else:
                config["out_sos_token"] = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
                config["out_eos_token"] = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
                config["out_pad_token"] = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]

        config["vocab_size"] = len(dataset.in_idx2word)
        config["symbol_size"] = len(dataset.out_idx2symbol)
        config["operator_nums"] = dataset.operator_nums
        config["copy_nums"] = dataset.copy_nums
        config["generate_size"] = len(dataset.generate_list)

        dataloader = create_dataloader(config)(config, dataset)
        if config["equation_fix"] == FixType.Prefix:
            evaluator = PreEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        elif config["equation_fix"] == FixType.Nonfix:
            evaluator = SeqEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        elif config["equation_fix"] == FixType.Postfix:
            evaluator = PostEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        else:
            raise NotImplementedError

        model = get_model(config["model"])(config).to(config["device"])
        if config["pretrained_model_path"]:
            config["vocab_size"] = len(model.tokenizer)
            config["embedding_size"] = len(model.tokenizer)
            config["in_word2idx"] = model.tokenizer.get_vocab()
            config["in_idx2word"] = list(model.tokenizer.get_vocab().keys())

        trainer = get_trainer(config["task_type"], config["model"])(config, model, dataloader, evaluator)
        if config["test only"]:
            NotImplementedError
        else:
            trainer.fit()


def run_toolkit():
    config = Config()
    print(config["device"])
    print(config["gpu_id"])
    init_seed(config['random_seed'], True)

    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)

    if config["k_fold"] != None:
        train_cross_validation(config)
    else:
        dataset.dataset_load()
        if config["share_vocab"]:
            config["out_symbol2idx"] = dataset.out_symbol2idx
            config["out_idx2symbol"] = dataset.out_idx2symbol
            config["in_word2idx"] = dataset.in_word2idx
            config["in_idx2word"] = dataset.in_idx2word
            config["out_sos_token"] = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            if config["symbol_for_tree"]:
                pass
            else:
                config["out_sos_token"] = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
                config["out_eos_token"] = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
                config["out_pad_token"] = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]

        config["vocab_size"] = len(dataset.in_idx2word)
        config["symbol_size"] = len(dataset.out_idx2symbol)
        config["operator_nums"] = dataset.operator_nums
        config["copy_nums"] = dataset.copy_nums
        config["generate_size"] = len(dataset.generate_list)

        dataloader = create_dataloader(config)(config, dataset)
        if config["equation_fix"] == FixType.Prefix:
            evaluator = PreEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        elif config["equation_fix"] == FixType.Nonfix:
            evaluator = SeqEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        elif config["equation_fix"] == FixType.Postfix:
            evaluator = PostEvaluater(dataset.out_symbol2idx, dataset.out_idx2symbol, config)
        else:
            raise NotImplementedError

        model = get_model(config["model"])(config).to(config["device"])
        if config["pretrained_model_path"]:
            config["vocab_size"] = len(model.tokenizer)
            config["embedding_size"] = len(model.tokenizer)
            config["in_word2idx"] = model.tokenizer.get_vocab()
            config["in_idx2word"] = list(model.tokenizer.get_vocab().keys())

        trainer = get_trainer(config["task_type"], config["model"])(config, model, dataloader, evaluator)
        if config["test_only"]:
            trainer.test()
        else:
            trainer.fit()
