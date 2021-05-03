import torch
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss

class MSELoss(AbstractLoss):
    _Name="mean squared error loss"
    def __init__(self):
        super().__init__(self._Name, F.mse_loss)
    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()
        loss /= self.norm_term
        return loss
    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1