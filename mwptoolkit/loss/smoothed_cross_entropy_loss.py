# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 18:55:01
# @File: smoothed_cross_entropy_loss.py


import torch
from torch import nn
from mwptoolkit.loss.abstract_loss import AbstractLoss


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -1, reduction: str = 'batchmean'):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        assert 0 < smoothing < 1, "Smoothing factor should be in (0.0, 1.0)"
        assert reduction in {'batchmean', 'none', 'sum'}
        super().__init__()

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Computes cross entropy loss with uniformly smoothed targets.
        Since the entropy of smoothed target distribution is always same, we can compute this with KL-divergence.

        :param torch.Tensor input: Log probability for each class. This is a Tensor with shape [B, C]
        :param torch.LongTensor target: List of target classes. This is a LongTensor with shape [B]
        :rtype: torch.Tensor
        :return: Computed loss
        """
        target = target.view(-1, 1)

        # Prepare smoothed target
        # Set all probability of the targets which should be ignored as zero.
        # Since D_KL(p, q) = p (log(p) - log(q)), by setting p(x) ≡ 0, these target cannot affect loss anymore.
        smoothed_target = torch.zeros(input.shape, requires_grad=False, device=target.device)

        # Set target values zero if predicted values are masked with -inf.
        for r, row in enumerate(input):
            tgt = target[r].item()
            if tgt == self.ignore_index:
                continue

            finites = torch.isfinite(row)
            n_cls = finites.sum().item()
            assert n_cls > 0

            smoothing_prob = self.smoothing / n_cls
            smoothed_target[r].masked_fill_(finites, smoothing_prob)
            smoothed_target[r, tgt] = 1.0 - self.smoothing

        # Compute loss: - p log q
        loss = - smoothed_target * input.masked_fill(~torch.isfinite(input), 0.0)

        if self.reduction == 'batchmean':
            return loss.sum() / input.shape[0]
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SmoothCrossEntropyLoss(AbstractLoss):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """
    _NAME = "SmoothCrossEntropyLoss"
    def __init__(self, weight=None, mask=None, size_average=True):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        super(SmoothCrossEntropyLoss, self).__init__(
            self._NAME,
            SmoothedCrossEntropyLoss())
        self.norm_term = 1

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()  # .data[0]
        
        loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 
        """
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term = 1
