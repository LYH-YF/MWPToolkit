from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.data.dataset.single_equation_dataset import SingleEquationDataset
from mwptoolkit.data.dataset.multi_equation_dataset import *

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.data.dataloader.single_equation_dataloader import SingleEquationDataLoader
def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    task_type = config['task_type'].lower()
    if task_type == "single_equation":
        return SingleEquationDataset(config) 
    else:
        return AbstractDataset(config)

def create_dataloader(config):
    task_type = config['task_type'].lower()
    if task_type == "single_equation":
        return SingleEquationDataLoader
    else:
        return AbstractDataLoader