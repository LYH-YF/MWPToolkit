# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:51:31
# @File: abstract_loss.py


from torch import nn


class AbstractLoss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        
        self.acc_loss = 0
        self.norm_term = 0

    def reset(self):
        """reset loss
        """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """return loss
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """calculate loss
        """
        raise NotImplementedError

    def backward(self):
        """loss backward
        """
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


