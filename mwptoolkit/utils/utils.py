# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:15:42
# @File: utils.py


import json
import math
import copy
import importlib
import random
import re
import numpy as np
import torch
from collections import OrderedDict

from mwptoolkit.utils.enum_type import TaskType,SupervisingMode


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
    """specially used to read data of ape200k source file
    """
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
    for i in l:
        if isinstance(i,list):
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


def time_since(s):
    """compute time

    Args:
        s (float): the amount of time in seconds.

    Returns:
        (str) : formatting time.
    """
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
        Model: model class
    """
    model_submodule = ['Seq2Seq', 'Seq2Tree', 'VAE', 'GAN', 'Graph2Tree','PreTrain']
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


def get_trainer_(task_type, model_name, sup_mode):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (~mwptoolkit.utils.enum_type.TaskType): model type
        model_name (str): model name

    Returns:
        ~mwptoolkit.trainer.trainer.Trainer: trainer class
    """
    if sup_mode == "fully_supervising":
        try:
            return getattr(importlib.import_module('mwptoolkit.trainer'),
                        model_name + 'Trainer')
        except AttributeError:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.supervised_trainer'),
                'SupervisedTrainer'
            )
    elif sup_mode == SupervisingMode.weakly_supervised:
        try: 
            return getattr(importlib.import_module('mwptoolkit.trainer.weakly_supervised_trainer'),
                        model_name + 'WeakTrainer')
        except AttributeError:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.weakly_supervised_trainer'),
                'WeaklySupervisedTrainer'
            )
    else:
        return getattr(
            importlib.import_module('mwptoolkit.trainer.abstract_trainer'),
            'AbstractTrainer'
        )

def get_trainer(config):
    r"""Automatically select trainer class based on task type and model name

    Args:
        config (~mwptoolkit.config.configuration.Config)

    Returns:
        ~mwptoolkit.trainer.SupervisedTrainer: trainer class
    """
    model_name = config["model"]
    sup_mode = config["supervising_mode"]
    if sup_mode == SupervisingMode.fully_supervised:
        if config['embedding']:
            try:
                return getattr(
                    importlib.import_module('mwptoolkit.trainer.supervised_trainer'),
                    'Pretrain' + model_name + 'Trainer'
                )
            except:
                if model_name.lower() in ['mathen']:
                    return getattr(
                        importlib.import_module('mwptoolkit.trainer.supervised_trainer'),
                        'PretrainSeq2SeqTrainer'
                    )
                else:
                    pass
        try:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.supervised_trainer'),
                model_name + 'Trainer'
            )
        except AttributeError:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.supervised_trainer'),
                'SupervisedTrainer'
            )

    elif sup_mode in SupervisingMode.weakly_supervised:
        try:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.weakly_supervised_trainer'),
                model_name + 'WeakTrainer'
            )
        except AttributeError:
            return getattr(
                importlib.import_module('mwptoolkit.trainer.weakly_supervised_trainer'),
                'WeaklySupervisedTrainer'
            )
    else:
        return getattr(
            importlib.import_module('mwptoolkit.trainer.abstract_trainer'),
            'AbstractTrainer'
        )


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


def clones(module, N):
    """Produce N identical layers.
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def str2float(v):
    """convert string to float.
    """
    if not isinstance(v,str):
        return v
    else:
        if '%' in v: # match %
            v=v[:-1]
            return float(v)/100
        if '(' in v:
            try:
                return eval(v) # match fraction
            except:
                if re.match('^\d+\(',v): # match fraction like '5(3/4)'
                    idx = v.index('(')
                    a = v[:idx]
                    b = v[idx:]
                    return eval(a)+eval(b)
                if re.match('.*\)\d+$',v): # match fraction like '(3/4)5'
                    l=len(v)
                    temp_v=v[::-1]
                    idx = temp_v.index(')')
                    a = v[:l-idx]
                    b = v[l-idx:]
                    return eval(a)+eval(b)
            return float(v)
        elif '/' in v: # match number like 3/4
            return eval(v)
        else:
            if v == '<UNK>':
                return float('inf')
            return float(v)


def lists2dict(list1,list2):
    r''' convert two lists to dict, elements of first list as keys, another's as values. 
    '''
    assert len(list1) == len(list2)
    the_dict=OrderedDict()
    for i,j in zip(list1,list2):
        the_dict[i]=j
    return the_dict

def get_weakly_supervised(supervising_mode):
    return getattr(importlib.import_module('mwptoolkit.module.Strategy.weakly_supervising'),
                   supervising_mode + 'Strategy')