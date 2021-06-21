import torch
from torch import nn
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss

class CrossEntropyLoss(AbstractLoss):
    _Name="cross entropy loss"
    def __init__(self,weight=None, mask=None, size_average=True):
        self.size_average=size_average
        super(CrossEntropyLoss,self).__init__(self._Name,F.cross_entropy)
    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()#.data[0]
        if self.size_average:
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        #print (outputs.size(), target.size())
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1