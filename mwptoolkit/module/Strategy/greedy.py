
# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:12:27
# @File: greedy.py


def greedy_search(logits):
    r"""Find the index of max logits

    Args:
        logits (torch.Tensor): logits distribution

    Return:
        torch.Tensor: the chosen index of token
    """
    return logits.argmax(dim=-1)
