import random
import torch
import warnings
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Encoder.rnn_encoder import GroupAttentionRNNEncoder
from mwptoolkit.module.Decoder.rnn_decoder import BasicRNNDecoder,AttentionalRNNDecoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder

class GroupATT(nn.Module):
    def __init__(self,config):
        super(GroupATT,self).__init__()
        self.bidirectional=config["bidirectional"]
        self.hidden_size=config["hidden_size"]
        self.encoder_rnn_cell_type=config["encoder_rnn_cell_type"]
        self.decoder_rnn_cell_type=config["decoder_rnn_cell_type"]
        self.attention=config["attention"]
        self.self_attention=config["self_attention"]
        self.share_vocab=config["share_vocab"]
        self.max_gen_len=config["max_output_len"]
        self.teacher_force_ratio=config["teacher_force_ratio"]
        self.num_start=config["num_start"]
        
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
        '''embedding_size, hidden_size,num_layers, bidirectional, rnn_cell_type, dropout_ratio,d_ff=2048,N=1'''
        self.encoder=GroupAttentionRNNEncoder(config["embedding_size"],config["hidden_size"],config["num_layers"],\
                                        config["bidirectional"],config["encoder_rnn_cell_type"],config["dropout_ratio"],config["in_word2idx"])
        
        if self.attention:
            self.decoder=AttentionalRNNDecoder(config["embedding_size"],config["decode_hidden_size"],config["hidden_size"]*2,\
                                                config["num_layers"],config["decoder_rnn_cell_type"],config["dropout_ratio"])
        else:
            self.decoder=BasicRNNDecoder(config["embedding_size"],config["decode_hidden_size"],config["num_layers"],\
                                            config["decoder_rnn_cell_type"],config["dropout_ratio"])
        
        self.dropout = nn.Dropout(config["dropout_ratio"])
        self.generate_linear = nn.Linear(config["decode_hidden_size"], config["symbol_size"])

    def forward(self,seq,seq_length,target=None):
        batch_size=seq.size(0)
        device=seq.device

        seq_emb=self.in_embedder(seq)
        encoder_outputs, encoder_hidden = self.encoder(seq,seq_emb, seq_length)
        
        # if self.bidirectional:
        #     encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
        
        # if self.bidirectional:
        #     if (self.encoder_rnn_cell_type == 'lstm'):
        #         encoder_hidden = (encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous())
        #     else:
        #         encoder_hidden = encoder_hidden[::2].contiguous()
        encoder_hidden  = tuple([self._cat_directions(h) for h in encoder_hidden])
        if self.encoder.rnn_cell_type == self.decoder.rnn_cell_type:
            pass
        elif (self.encoder.rnn_cell_type == 'gru') and (self.decoder.rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder.rnn_cell_type == 'rnn') and (self.decoder.rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder.rnn_cell_type == 'lstm') and (self.decoder.rnn_cell_type == 'gru' or self.decoder.rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass

        decoder_inputs=self.init_decoder_inputs(target,device,batch_size)
        if target!=None:
            token_logits=self.generate_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return token_logits
        else:
            all_outputs=self.generate_without_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return all_outputs

    def init_decoder_inputs(self,target,device,batch_size):
        pad_var = torch.LongTensor([self.out_sos_token]*batch_size).to(device).view(batch_size,1)
        if target != None:
            decoder_inputs=torch.cat((pad_var,target),dim=1)[:,:-1]
        else:
            decoder_inputs=pad_var
        decoder_inputs=self.out_embedder(decoder_inputs)
        return decoder_inputs
    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
    def decode(self,output):
        device=output.device

        batch_size=output.size(0)
        decoded_output=[]
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output=torch.tensor(decoded_output).to(device).view(batch_size,-1)
        return output
    
    def rule1_filter(self):
        r"""if r_t−1 in {+, −, ∗, /}, then rt will not in {+, −, ∗, /,), =}.
        """
        filters = []
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        filters.append(self.out_symbol2idx['<EOS>'])
        return torch.tensor(filters).long()
    def rule2_filter(self):
        r"""if r_t-1 is a number, then r_t will not be a number and not in {(, =)}.
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx['('])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        for idx in range(self.num_start,len(self.out_idx2symbol)):
            filters.append(idx)
        return torch.tensor(filters).long()
    def rule3_filter(self):
        r"""if rt−1 is '=', then rt will not in {+, −, ∗, /, =,)}.
        """
        filters = []
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        return torch.tensor(filters).long()
    def rule4_filter(self):
        r"""if r_t-1 is '(' , then r_t will not in {(,), +, -, *, /, =}).
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx['('])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        filters.append(self.out_symbol2idx['<EOS>'])
        return torch.tensor(filters).long()
    def rule5_filter(self):
        r"""if r_t−1 is ')', then r_t will not be a number and not in {(,)};
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx['('])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        for idx in range(self.num_start,len(self.out_idx2symbol)):
            filters.append(idx)
        return torch.tensor(filters).long()
    def filter_op(self):
        filters = []
        filters.append(self.out_symbol2idx['+']) 
        filters.append(self.out_symbol2idx['-']) 
        filters.append(self.out_symbol2idx['*']) 
        filters.append(self.out_symbol2idx['/']) 
        filters.append(self.out_symbol2idx['^']) 
        return torch.tensor(filters).long()
    def filter_END(self):
        filters = []
        filters.append(self.out_symbol2idx['<EOS>']) 
        return torch.tensor(filters).long()
    
    def rule_filter_(self,symbols,token_logit):
        r"""
        Args:
            symbols: torch.Tensor, [batch_size]
            token_logit: torch.Tensor, [batch_size, symbol_size]
        return:
            symbols of next step : [batch_size]
        """
        device=token_logit.device
        next_symbols=[]
        current_logit=token_logit.clone().detach()
        if symbols==[]:
            filters=torch.cat([self.filter_op(),self.filter_END()])
            for b in range(current_logit.size(0)):
                current_logit[b][filters]=-float('inf')
        else:
            for b,symbol in enumerate(symbols.split(1)):
                if self.out_idx2symbol[symbol] in ['+','-','*','/','^']:
                    filters = self.rule1_filter()
                    current_logit[b][filters] = -float('inf')
                elif symbol >= self.num_start:
                    filters = self.rule2_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in ['=']:
                    filters = self.rule3_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in ['(']:
                    filters = self.rule4_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in [')']:
                    filters = self.rule5_filter()
                    current_logit[b][filters] = -float('inf')
        next_symbols = current_logit.topk(1,dim=1)[1]
        return next_symbols
    
    def __str__(self):
        info=super().__str__()
        total=sum(p.numel() for p in self.parameters())
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters="\ntotal parameters : {} \ntrainable parameters : {}".format(total,trainable)
        return info+parameters
    
    def generate_t(self,encoder_outputs,encoder_hidden,decoder_inputs):
        with_t=random.random()
        if with_t<self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, encoder_hidden,encoder_outputs)
            else:
                decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            token_logits=token_logits.view(-1, token_logits.size(-1))
            token_logits=torch.nn.functional.log_softmax(token_logits,dim=1)
            # token_logits=torch.log_softmax(token_logits,dim=1)
        else:
            seq_len=decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:,0,:].unsqueeze(1)
            token_logits=[]
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
                #attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                predict=torch.nn.functional.log_softmax(token_logit,dim=1)
                #predict=torch.log_softmax(token_logit,dim=1)
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
            if self.attention:
                decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
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
    
    # def generate_t(self,encoder_outputs,encoder_hidden,decoder_inputs):
    #     with_t=random.random()
    #     seq_len=decoder_inputs.size(1)
    #     decoder_hidden = encoder_hidden
    #     decoder_input = decoder_inputs[:,0,:].unsqueeze(1)
    #     token_logits=[]
    #     output=[]
    #     for idx in range(seq_len):
    #         if with_t<self.teacher_force_ratio:
    #             decoder_input = decoder_inputs[:,idx,:].unsqueeze(1)
    #         if self.attention:
    #             decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
    #         else:
    #             decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
    #         #attn_list.append(attn)
    #         step_output = decoder_output.squeeze(1)
    #         token_logit = self.generate_linear(step_output)
    #         predict=torch.nn.functional.log_softmax(token_logit,dim=1)
    #         output=self.rule_filter_(output,token_logit)
    #         token_logits.append(predict)

    #         if self.share_vocab:
    #             output_=self.decode(output)
    #             decoder_input=self.out_embedder(output_)
    #         else:
    #             decoder_input=self.out_embedder(output)
    #     token_logits=torch.stack(token_logits,dim=1)
    #     token_logits=token_logits.view(-1,token_logits.size(-1))
    #     return token_logits
    
    # def generate_without_t(self,encoder_outputs,encoder_hidden,decoder_input):
    #     all_outputs=[]
    #     decoder_hidden = encoder_hidden
    #     output=[]
    #     for idx in range(self.max_gen_len):
    #         if self.attention:
    #             decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
    #         else:
    #             decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            
    #         step_output = decoder_output.squeeze(1)
    #         token_logits = self.generate_linear(step_output)
    #         output = self.rule_filter_(output,token_logits)
            
    #         all_outputs.append(output)
    #         if self.share_vocab:
    #             output_=self.decode(output)
    #             decoder_input=self.out_embedder(output_)
    #         else:
    #             decoder_input=self.out_embedder(output)
    #     all_outputs=torch.cat(all_outputs,dim=1)
    #     return all_outputs
    