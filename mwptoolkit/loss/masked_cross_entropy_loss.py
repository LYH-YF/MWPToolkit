# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:54:00
# @File: masked_cross_entropy_loss.py


import torch
from torch.nn import functional
from mwptoolkit.loss.abstract_loss import AbstractLoss


def masked_cross_entropy(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        mask: A Variable for target containing a BoolTensor of size (batch, max_len)
    Returns:
        loss: An loss value.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask=mask.float()
    
    losses = losses * mask    
    loss=losses.sum()/mask.sum()
    
    return loss

class MaskedCrossEntropyLoss(AbstractLoss):
    _Name="avg masked cross entopy loss"
    def __init__(self):
        super().__init__(self._Name, masked_cross_entropy)

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()#.data[0]
        return loss
    
    def eval_batch(self, outputs, target, mask):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 

            mask (Tensor): mask to ignore loss.
        """
        self.acc_loss += self.criterion(outputs, target,mask)
        self.norm_term += 1