from torch import nn


class AbstractLoss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        
        self.acc_loss = 0
        self.norm_term = 0

    def reset(self):
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        raise NotImplementedError

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


