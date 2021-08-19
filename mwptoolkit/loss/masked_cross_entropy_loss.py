# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/19 10:03:26
# @File: masked_cross_entropy_loss.py


import torch
from torch.nn import functional as F
from mwptoolkit.loss.abstract_loss import AbstractLoss


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


# def masked_cross_entropy(logits, target, mask):
#     """
#     Args:
#         logits: A Variable containing a FloatTensor of size
#             (batch, max_len, num_classes) which contains the
#             unnormalized probability for each class.
#         target: A Variable containing a LongTensor of size
#             (batch, max_len) which contains the index of the true
#             class for each corresponding step.
#         mask: A Variable for target containing a BoolTensor of size (batch, max_len)
#     Returns:
#         loss: An loss value.
#     """

#     # logits_flat: (batch * max_len, num_classes)
#     logits_flat = logits.view(-1, logits.size(-1))
#     # log_probs_flat: (batch * max_len, num_classes)
#     log_probs_flat = functional.log_softmax(logits_flat, dim=1)
#     # target_flat: (batch * max_len, 1)
#     target_flat = target.view(-1, 1)
#     # losses_flat: (batch * max_len, 1)
#     losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

#     # losses: (batch, max_len)
#     losses = losses_flat.view(*target.size())
#     # mask: (batch, max_len)
#     mask=mask.float()

#     losses = losses * mask
#     loss=losses.sum()/mask.sum()

#     return loss


class MaskedCrossEntropyLoss(AbstractLoss):
    _Name = "avg masked cross entopy loss"

    def __init__(self):
        super().__init__(self._Name, masked_cross_entropy)

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()  #.data[0]
        return loss

    def eval_batch(self, outputs, target, length):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 

            length (Tensor): length of target.
        """
        self.acc_loss += self.criterion(outputs, target, length)
        self.norm_term += 1