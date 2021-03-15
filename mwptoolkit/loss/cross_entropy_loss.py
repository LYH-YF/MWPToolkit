import torch
from torch import nn
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss

class CrossEntropyLoss(AbstractLoss):
    _Name="cross entropy loss"
    def __init__(self,weight=None, mask=None, size_average=True):
        super().__init__(self._Name,F.cross_entropy)
        nn.CrossEntropyLoss
        pass