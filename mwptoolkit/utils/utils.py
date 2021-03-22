import json
import math
import importlib
import random
import numpy as np
import torch

from mwptoolkit.utils.enum_type import TaskType


def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


def read_json_data(filename):
    '''
    load data from a json file
    '''
    f = open(filename, 'r', encoding="utf-8")
    return json.load(f)


def read_ape200k_source(filename):
    data_list = []
    f = open(filename, 'r', encoding="utf-8")
    for line in f:
        data_list.append(json.loads(line))
    return data_list


def read_math23k_source(filename):
    """
    specially used to read data of math23k source file
    """
    data_list = []
    f = open(filename, 'r', encoding="utf-8")
    count = 0
    string = ''
    for line in f:
        count += 1
        string += line
        if count % 7 == 0:
            data_list.append(json.loads(string))
            string = ''
    return data_list


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Generator: model class
    """
    model_submodule = ['Seq2Seq', 'Seq2Tree', 'VAE', 'GAN', 'Graph2Tree']
    try:
        model_file_name = model_name.lower()
        for submodule in model_submodule:
            module_path = '.'.join(['...model', submodule, model_file_name])
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)

        model_class = getattr(model_module, model_name)
    except:
        raise NotImplementedError("{} can't be found".format(model_file_name))
    return model_class


def get_trainer(task_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (~mwptoolkit.utils.enum_type.TaskType): model type
        model_name (str): model name

    Returns:
        ~mwptoolkit.trainer.trainer.Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('mwptoolkit.trainer.trainer'),
                       model_name + 'Trainer')
    except AttributeError:
        if task_type == TaskType.SingleEquation:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.trainer'),
                'SingleEquationTrainer')
        elif task_type == TaskType.MultiEquation:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.trainer'),
                'MultiEquationTrainer')
        else:
            return getattr(importlib.import_module('mwptoolkit.trainer.trainer'),
                           'Trainer')


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False