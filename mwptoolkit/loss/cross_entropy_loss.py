# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:53:39
# @File: cross_entropy_loss.py


import torch
from torch import nn
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss

class CrossEntropyLoss(AbstractLoss):
    _Name="cross entropy loss"
    def __init__(self,weight=None, mask=None, size_average=True):
        """
        Args:
            weight (Tensor, optional): a manual rescaling weight given to each class.
            
            mask (Tensor, optional): index of classes to rescale weight
        """
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is not None:
                weight[mask] = 0
        super(CrossEntropyLoss,self).__init__(
            self._Name,
            nn.CrossEntropyLoss(weight=weight,reduction='mean')
        )
    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()#.data[0]
        if self.size_average:
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 
        """
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1