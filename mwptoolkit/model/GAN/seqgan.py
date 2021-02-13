import torch
from torch import nn

from mwptoolkit.module.Generator.seqgan_generator import SeqGANGenerator
from mwptoolkit.module.Discriminator.seqgan_discriminator import SeqGANDiscriminator

class SeqGAN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.generator=SeqGANGenerator(config)
        self.discriminator=SeqGANDiscriminator(config)
    def forward(self,seq,seq_length,target=None):
        all_output,token_logits,_,_=self.generator.forward(seq,seq_length,target)
        if target != None:
            return token_logits
        else:
            return all_output