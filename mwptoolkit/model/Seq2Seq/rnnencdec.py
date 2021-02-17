import random
import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Decoder.rnn_decoder import BasicRNNDecoder,AttentionalRNNDecoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder

class RNNEncDec(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.bidirectional=config["bidirectional"]
        self.hidden_size=config["hidden_size"]
        self.rnn_cell_type=config["rnn_cell_type"]
        self.attention=config["attention"]
        self.share_vocab=config["share_vocab"]
        self.max_gen_len=30
        self.teacher_force_ratio=0.9
        if config["share_vocab"]:
            self.out_symbol2idx=config["out_symbol2idx"]
            self.out_idx2symbol=config["out_idx2symbol"]
            self.in_word2idx=config["in_word2idx"]
            self.in_idx2word=config["in_idx2word"]
            self.out_sos_token=config["out_sos_token"]
        else:
            self.out_sos_token=config["out_sos_token"]

        self.in_embedder=BaiscEmbedder(config["vocab_size"],config["embedding_size"],config["dropout_ratio"])
        if config["share_vocab"]:
            self.out_embedder=self.in_embedder
        else:
            self.out_embedder=BaiscEmbedder(config["symbol_size"],config["embedding_size"],config["dropout_ratio"])

        self.encoder=BasicRNNEncoder(config["embedding_size"],config["hidden_size"],config["num_layers"],\
                                        config["rnn_cell_type"],config["dropout_ratio"])
        if self.attention:
            self.decoder=AttentionalRNNDecoder(config["embedding_size"],config["decode_hidden_size"],config["hidden_size"],\
                                                config["num_layers"],config["rnn_cell_type"],config["dropout_ratio"])
        else:
            self.decoder=BasicRNNDecoder(config["embedding_size"],config["decode_hidden_size"],config["num_layers"],\
                                            config["rnn_cell_type"],config["dropout_ratio"])
        
        self.dropout = nn.Dropout(config["dropout_ratio"])
        self.generate_linear = nn.Linear(config["hidden_size"], config["symbol_size"])

    def forward(self,seq,seq_length,target=None):
        batch_size=seq.size(0)
        device=seq.device

        seq_emb=self.in_embedder(seq)
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                encoder_hidden = (encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous())
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        decoder_inputs=self.init_decoder_inputs(target,device,batch_size)
        if target!=None:
            token_logits=self.generate_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return token_logits
        else:
            all_outputs=self.generate_without_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return all_outputs

    def generate_t(self,encoder_outputs,encoder_hidden,decoder_inputs):
        with_t=random.random()
        if with_t<self.teacher_force_ratio:
            decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden,encoder_outputs)
            token_logits = self.generate_linear(decoder_outputs)
            token_logits=token_logits.view(-1, token_logits.size(-1))
            #token_logits=torch.nn.functional.log_softmax(token_logits,dim=1)
            token_logits=torch.log_softmax(token_logits,dim=1)
        else:
            seq_len=decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:,0,:].unsqueeze(1)
            token_logits=[]
            for idx in range(seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
                #attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                #predict=torch.nn.functional.log_softmax(token_logit,dim=1)
                predict=torch.log_softmax(token_logits,dim=1)
                output=predict.topk(1,dim=1)[1]
                token_logits.append(predict)

                if self.share_vocab:
                    output=self.decode(output)
                    decoder_input=self.out_embedder(output)
                else:
                    decoder_input=self.out_embedder(output)
            token_logits=torch.stack(token_logits,dim=1)
            token_logits=token_logits.view(-1,token_logits.size(-1))
        return token_logits
    
    def generate_without_t(self,encoder_outputs,encoder_hidden,decoder_input):
        all_outputs=[]
        decoder_hidden = encoder_hidden
        for idx in range(self.max_gen_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            token_logits = self.generate_linear(step_output)
            predict=torch.nn.functional.log_softmax(token_logits,dim=1)
            output=predict.topk(1,dim=1)[1]
            
            
            all_outputs.append(output)
            if self.share_vocab:
                output=self.decode(output)
                decoder_input=self.out_embedder(output)
            else:
                decoder_input=self.out_embedder(output)
        all_outputs=torch.cat(all_outputs,dim=1)
        return all_outputs
    
    def init_decoder_inputs(self,target,device,batch_size):
        pad_var = torch.LongTensor([self.out_sos_token]*batch_size).to(device).view(batch_size,1)
        if target != None:
            decoder_inputs=torch.cat((pad_var,target),dim=1)[:,:-1]
        else:
            decoder_inputs=pad_var
        decoder_inputs=self.out_embedder(decoder_inputs)
        return decoder_inputs
    def decode(self,output):
        device=output.device

        batch_size=output.size(0)
        decoded_output=[]
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output=torch.tensor(decoded_output).to(device).view(batch_size,-1)
        return output
