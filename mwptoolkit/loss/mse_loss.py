# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:54:15
# @File: mse_loss.py


import torch
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss

class MSELoss(AbstractLoss):
    _Name="mean squared error loss"
    def __init__(self):
        super().__init__(self._Name, F.mse_loss)
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