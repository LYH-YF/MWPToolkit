# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:35:57
# @File: template_dataloader.py


import torch

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.utils.enum_type import FixType, NumMask

class TemplateDataLoader(AbstractDataLoader):
    """template dataloader.

    you need implement:

    TemplateDataLoader.load_batch
    
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
    
    def load_data(self, type):
        if type == "train":
            datas = self.dataset.trainset
            batch_size = self.train_batch_size
        elif type == "valid":
            datas = self.dataset.validset
            batch_size = self.test_batch_size
        elif type == "test":
            datas = self.dataset.testset
            batch_size = self.test_batch_size
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))

        num_total = len(datas)
        batch_num = int(num_total / batch_size) + 1
        for batch_i in range(batch_num):
            start_idx = batch_i * batch_size
            end_idx = (batch_i + 1) * batch_size
            if end_idx <= num_total:
                batch_data = datas[start_idx:end_idx]
            else:
                batch_data = datas[start_idx:num_total]
            if batch_data != []:
                batch_data = self.load_batch(batch_data)
                yield batch_data
    
    def load_batch(self,batch):
        raise NotImplementedError