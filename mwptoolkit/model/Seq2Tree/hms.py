import torch
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import HWCPEncoder
from mwptoolkit.module.Decoder.tree_decoder import HMSDecoder

class HMS(nn.Module):
    def __init__(self,config):
        super(HMS,self).__init__()
        self.embedder=BaiscEmbedder(config['vocab_size'],config['embedding_size'],config['dropout_ratio'])
        self.encoder=HWCPEncoder(self.embedder,config['embedding_size'],config['hidden_size'],config['span_size'],config['dropout_ratio'])
        self.decoder=HMSDecoder(self.embedder,config['hidden_size'],config['dropout_ratio'],config['operator_list'],config['in_word2idx'],config['out_idx2symbol'],config["device"])
    def forward(self, input_variable, input_lengths,span_num_pos,word_num_poses, span_length=None,tree=None,
                target_variable=None, max_length=None, beam_width=None):
        num_pos=(span_num_pos,word_num_poses)
        encoder_outputs, encoder_hidden = self.encoder(
            input_var=input_variable,
            input_lengths=input_lengths,
            span_length=span_length,
            tree=tree
        )

        output = self.decoder(
            targets=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            max_length=max_length,
            beam_width=beam_width
        )
        return output