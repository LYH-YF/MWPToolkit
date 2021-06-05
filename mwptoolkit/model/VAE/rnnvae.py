import random
import torch
from torch import nn

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Decoder.rnn_decoder import BasicRNNDecoder,AttentionalRNNDecoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Strategy.sampling import topk_sampling
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.utils.enum_type import SpecialTokens,NumMask

class RNNVAE(nn.Module):
    def __init__(self, config,dataset):
        super(RNNVAE, self).__init__()

        # load parameters info
        self.max_length = config["max_output_len"]
        self.share_vocab=config["share_vocab"]

        self.num_directions = 2 if config['bidirectional'] else 1
        self.rnn_cell_type=config['rnn_cell_type']
        self.bidirectional=config["bidirectional"]
        self.attention=config["attention"]
        self.embedding_size=config["embedding_size"]
        self.latent_size = config['latent_size']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.hidden_size=config["hidden_size"]
        self.teacher_force_ratio=config['teacher_force_ratio']
        self.dropout_ratio=config["dropout_ratio"]

        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.padding_token_idx = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        # define layers and loss
        self.in_embedder = BaiscEmbedder(self.vocab_size,self.embedding_size,self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder=self.in_embedder
        else:
            self.out_embedder=BaiscEmbedder(self.symbol_size,self.embedding_size,self.dropout_ratio)

        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_encoder_layers, self.rnn_cell_type,
                                       self.dropout_ratio, self.bidirectional)
        if self.attention:
            self.decoder=AttentionalRNNDecoder(self.embedding_size+self.latent_size,self.hidden_size,self.hidden_size,\
                                                self.num_decoder_layers,self.rnn_cell_type,self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size+self.latent_size, self.hidden_size, self.num_decoder_layers, self.rnn_cell_type,
                                       self.dropout_ratio)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.out = nn.Linear(self.hidden_size, self.symbol_size)

        if self.rnn_cell_type == "lstm":
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(self.latent_size, 2 * self.hidden_size)
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(self.latent_size, 2 * self.hidden_size)
        else:
            raise ValueError("No such rnn type {} for RNNVAE.".format(self.rnn_cell_type))

        weight = torch.ones(self.symbol_size).to(config["device"])
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq,seq_length,target=None):
        batch_size = seq.size(0)
        device=seq.device

        input_emb = self.in_embedder(seq)
        encoder_outputs, hidden_states = self.encoder(input_emb, seq_length)

        if self.rnn_cell_type == "lstm":
            h_n, c_n = hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError("No such rnn type {} for RNNVAE.".format(self.rnn_cell_type))

        if self.bidirectional:
            h_n = h_n.view(self.num_encoder_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        z = torch.randn([batch_size, self.latent_size]).to(device)
        z = mean + z * torch.exp(0.5 * logvar)

        #hidden = self.latent_to_hidden(z)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                hidden_states = (hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous())
            else:
                hidden_states = hidden_states[::2].contiguous()

        # if self.rnn_cell_type == "lstm":
        #     decoder_hidden = torch.chunk(hidden, 2, dim=-1)
        #     h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        #     c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        #     decoder_hidden = (h_0, c_0)
        # else:
        #     decoder_hidden = hidden.unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        
        decoder_inputs=self.init_decoder_inputs(target,device,batch_size)
        if target!=None:
            token_logits=self.generate_t(encoder_outputs,decoder_inputs,hidden_states,z)
            return token_logits
        else:
            all_outputs=self.generate_without_t(encoder_outputs,decoder_inputs,hidden_states,z)
            return all_outputs
    
    def calculate_loss(self,batch_data):
        seq = batch_data['question']
        seq_length = batch_data['ques len']
        target = batch_data['equation']

        batch_size = seq.size(0)
        device=seq.device

        input_emb = self.in_embedder(seq)
        encoder_outputs, hidden_states = self.encoder(input_emb, seq_length)

        if self.rnn_cell_type == "lstm":
            h_n, c_n = hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError("No such rnn type {} for RNNVAE.".format(self.rnn_cell_type))

        if self.bidirectional:
            h_n = h_n.view(self.num_encoder_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        z = torch.randn([batch_size, self.latent_size]).to(device)
        z = mean + z * torch.exp(0.5 * logvar)

        #hidden = self.latent_to_hidden(z)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                hidden_states = (hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous())
            else:
                hidden_states = hidden_states[::2].contiguous()
        
        decoder_inputs=self.init_decoder_inputs(target,device,batch_size)

        token_logits=self.generate_t(encoder_outputs,decoder_inputs,hidden_states,z)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        self.loss.reset()
        self.loss.eval_batch(token_logits, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self,batch_data):
        seq = batch_data['question']
        seq_length = batch_data['ques len']
        target = batch_data['equation']
        num_list = batch_data['num list']

        batch_size = seq.size(0)
        device=seq.device

        input_emb = self.in_embedder(seq)
        encoder_outputs, hidden_states = self.encoder(input_emb, seq_length)

        if self.rnn_cell_type == "lstm":
            h_n, c_n = hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError("No such rnn type {} for RNNVAE.".format(self.rnn_cell_type))

        if self.bidirectional:
            h_n = h_n.view(self.num_encoder_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        z = torch.randn([batch_size, self.latent_size]).to(device)
        z = mean + z * torch.exp(0.5 * logvar)

        #hidden = self.latent_to_hidden(z)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.rnn_cell_type == 'lstm'):
                hidden_states = (hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous())
            else:
                hidden_states = hidden_states[::2].contiguous()
        
        decoder_inputs=self.init_decoder_inputs(None,device,batch_size)

        all_outputs=self.generate_without_t(encoder_outputs,decoder_inputs,hidden_states,z)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(all_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def generate_t(self,encoder_outputs,decoder_inputs,decoder_hidden,z):
        with_t=random.random()
        if with_t<self.teacher_force_ratio:
            decoder_inputs=torch.cat((decoder_inputs,z.unsqueeze(1).repeat(1,decoder_inputs.size(1),1)),dim=2)
            decoder_output, hidden_states = self.decoder(input_embeddings=decoder_inputs, hidden_states=decoder_hidden,encoder_outputs=encoder_outputs)
            token_logits = self.out(decoder_output)
            token_logits=token_logits.view(-1, token_logits.size(-1))
            token_logits=torch.nn.functional.log_softmax(token_logits,dim=1)
        else:
            seq_len=decoder_inputs.size(1)
            decoder_input = decoder_inputs[:,0,:].unsqueeze(1)
            decoder_input=torch.cat((decoder_input,z.unsqueeze(1).repeat(1,decoder_input.size(1),1)),dim=2)
            token_logits=[]
            for idx in range(seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
                #step_output = decoder_output.squeeze(1)
                decoder_output = decoder_output.squeeze(1)
                token_logit = self.out(decoder_output)
                predict=torch.nn.functional.log_softmax(token_logit,dim=1)
                #output=topk_sampling(predict)
                output=predict.topk(1,dim=1)[1]
                token_logits.append(predict)

                if self.share_vocab:
                    output=self.convert_out_idx_2_in_idx(output)
                    decoder_input=self.out_embedder(output)
                else:
                    decoder_input=self.out_embedder(output)
                decoder_input=torch.cat((decoder_input,z.unsqueeze(1).repeat(1,decoder_input.size(1),1)),dim=2)
            token_logits=torch.stack(token_logits,dim=1)
            token_logits=token_logits.view(-1,token_logits.size(-1))
        return token_logits
    
    def generate_without_t(self,encoder_outputs,decoder_input,decoder_hidden,z):
        all_outputs=[]
        for _ in range(self.max_length):
            decoder_input=torch.cat((decoder_input,z.unsqueeze(1).repeat(1,decoder_input.size(1),1)),dim=2)
            decoder_output, decoder_hidden = self.decoder(input_embeddings=decoder_input, hidden_states=decoder_hidden,encoder_outputs=encoder_outputs)
            decoder_output = decoder_output.squeeze(1)
            token_logits = self.out(decoder_output)
            predict=torch.nn.functional.log_softmax(token_logits,dim=1)
            #output = topk_sampling(predict)
            output=predict.topk(1,dim=1)[1]
            
            all_outputs.append(output)
            if self.share_vocab:
                output=self.convert_out_idx_2_in_idx(output)
                decoder_input=self.out_embedder(output)
            else:
                decoder_input=self.out_embedder(output)
        all_outputs=torch.cat(all_outputs,dim=1)
        return all_outputs
    
    def init_decoder_inputs(self,target,device,batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx]*batch_size).to(device).view(batch_size,1)
        if target != None:
            decoder_inputs=torch.cat((pad_var,target),dim=1)[:,:-1]
        else:
            decoder_inputs=pad_var
        decoder_inputs=self.out_embedder(decoder_inputs)
        return decoder_inputs
    
    def convert_out_idx_2_in_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


    def decode(self,output):
        device=output.device
        
        batch_size=output.size(0)
        decoded_output=[]
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output=torch.tensor(decoded_output).to(device).view(batch_size,-1)
        return output