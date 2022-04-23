# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:39:08
# @File: utils.py
from typing import Union, Type

from mwptoolkit.config.configuration import Config

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.data.dataset.single_equation_dataset import SingleEquationDataset
from mwptoolkit.data.dataset.multi_equation_dataset import MultiEquationDataset
from mwptoolkit.data.dataset.dataset_multiencdec import DatasetMultiEncDec
from mwptoolkit.data.dataset.dataset_ept import DatasetEPT
from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset
from mwptoolkit.data.dataset.dataset_hms import DatasetHMS
from mwptoolkit.data.dataset.dataset_gpt2 import DatasetGPT2

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.data.dataloader.single_equation_dataloader import SingleEquationDataLoader
from mwptoolkit.data.dataloader.multi_equation_dataloader import MultiEquationDataLoader
from mwptoolkit.data.dataloader.dataloader_multiencdec import DataLoaderMultiEncDec
from mwptoolkit.data.dataloader.dataloader_ept import DataLoaderEPT
from mwptoolkit.data.dataloader.pretrain_dataloader import PretrainDataLoader
from mwptoolkit.data.dataloader.dataloader_hms import DataLoaderHMS
from mwptoolkit.data.dataloader.dataloader_gpt2 import DataLoaderGPT2

from mwptoolkit.utils.enum_type import TaskType


def create_dataset(config):
    """Create dataset according to config

    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    try:
        return eval('Dataset{}'.format(config['model']))(config)
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataset(config)
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataset(config)
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataset(config)
    else:
        return AbstractDataset(config)


def create_dataloader(config):
    """Create dataloader according to config

    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataloader module
    """
    try:
        return eval('DataLoader{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataLoader
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataLoader
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataLoader
    else:
        return AbstractDataLoader


def get_dataset_module(config: Config) \
        -> Type[Union[
            DatasetMultiEncDec, DatasetEPT, DatasetHMS, DatasetGPT2, PretrainDataset, SingleEquationDataset, MultiEquationDataset, AbstractDataset]]:
    """
    return a dataset module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: dataset module
    """
    try:
        return eval('Dataset{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataset
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataset
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataset
    else:
        return AbstractDataset


def get_dataloader_module(config: Config) \
        -> Type[Union[
            DataLoaderMultiEncDec, DataLoaderEPT, DataLoaderHMS, DataLoaderGPT2, PretrainDataLoader, SingleEquationDataLoader, MultiEquationDataLoader, AbstractDataLoader]]:
    """Create dataloader according to config

        Args:
            config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

        Returns:
            Dataloader module
        """
    try:
        return eval('DataLoader{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataLoader
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataLoader
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataLoader
    else:
        return AbstractDataLoader
