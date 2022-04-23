# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2022/2/18 12:40
# @File: abstract_model.py
# @Update Time: 2022/2/18 12:40
from typing import Union
from functools import singledispatch

import torch
from torch import nn

from mwptoolkit.config.configuration import Config

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
# from mwptoolkit.data.dataset.dataset_ept import DatasetEPT
# from mwptoolkit.data.dataset.dataset_hms import DatasetHMS
from mwptoolkit.data.dataset.template_dataset import TemplateDataset
from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset
from mwptoolkit.data.dataset.dataset_multiencdec import DatasetMultiEncDec
from mwptoolkit.data.dataset.multi_equation_dataset import MultiEquationDataset
from mwptoolkit.data.dataset.single_equation_dataset import SingleEquationDataset


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, batch_data: dict):
        raise NotImplementedError

    def model_test(self, batch_data: dict):
        raise NotImplementedError

    def predict(self, batch_data: dict, output_all_layers: bool = False):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError

    def save_model(self, trained_dir):
        raise NotImplementedError

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters
