# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:52:44
# @File: binary_cross_entropy_loss.py


import torch
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss


class BinaryCrossEntropyLoss(AbstractLoss):
    _Name = "binary cross entropy loss"

    def __init__(self):
        super().__init__(self._Name, F.binary_cross_entropy)

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()
        loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target distribution. 
        """
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

    def add_norm(self, norm):
        self.acc_loss /= self.norm_term
        self.acc_loss += norm
        self.norm_term = 1
