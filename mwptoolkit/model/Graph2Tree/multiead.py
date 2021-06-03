import torch
from torch import nn

class MultiEAD(nn.Module):
    def __init__(self,config):
        super().__init__()