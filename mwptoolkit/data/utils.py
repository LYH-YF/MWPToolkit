from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.data.dataset.single_equation_dataset import SingleEquationDataset
from mwptoolkit.data.dataset.multi_equation_dataset import MultiEquationDataset
from mwptoolkit.data.dataset.dataset_multiencdec import DatasetMultiEncDec
from mwptoolkit.data.dataset.dataset_ept import DatasetEPT
from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.data.dataloader.single_equation_dataloader import SingleEquationDataLoader
from mwptoolkit.data.dataloader.multi_equation_dataloader import MultiEquationDataLoader
from mwptoolkit.data.dataloader.dataloader_multiencdec import DataLoaderMultiEncDec
from mwptoolkit.data.dataloader.dataloader_ept import DataLoaderEPT
from mwptoolkit.data.dataloader.pretrain_dataloader import PretrainDataLoader
from mwptoolkit.utils.enum_type import TaskType

def create_dataset(config):
    """Create dataset according to config

    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    if config['model'].lower() in ['multiencdec']:
        return DatasetMultiEncDec(config)
    if config['model'].lower() in ['ept']:
        return DatasetEPT(config)
    if config['embedding'] != None:
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
    if config['model'].lower() in ['multiencdec']:
        return DataLoaderMultiEncDec
    if config['model'].lower() in ['ept']:
        return DataLoaderEPT
    if config['embedding'] != None:
        return PretrainDataLoader
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataLoader
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataLoader
    else:
        return AbstractDataLoader