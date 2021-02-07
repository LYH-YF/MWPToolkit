import random
import torch
from torch import nn

from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Decoder.rnn_decoder import BasicRNNDecoder,AttentionalRNNDecoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Strategy.sampling import topk_sampling

class RNNVAE(nn.Module):
    def __init__(self, config):
        super(RNNVAE, self).__init__()

        # load parameters info
        self.max_length = config["max_output_len"]

        self.num_directions = 2 if config['bidirectional'] else 1
        self.rnn_cell_type=config['rnn_cell_type']
        self.bidirectional=config["bidirectional"]
        self.attention=config["attention"]
        
        self.latent_size = config['latent_size']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.hidden_size=config["hidden_size"]
        
        self.padding_token_idx = config["out_pad_token"]
        self.share_vocab=config["share_vocab"]
        self.teacher_force_ratio=0.9
        if config["share_vocab"]:
            self.out_symbol2idx=config["out_symbol2idx"]
            self.out_idx2symbol=config["out_idx2symbol"]
            self.in_word2idx=config["in_word2idx"]
            self.in_idx2word=config["in_idx2word"]
            self.out_sos_token=config["out_sos_token"]
        else:
            self.out_sos_token=config["out_sos_token"]

        # define layers and loss
        self.in_embedder = BaiscEmbedder(config["vocab_size"],config["embedding_size"],config["dropout_ratio"])
        if config["share_vocab"]:
            self.out_embedder=self.in_embedder
        else:
            self.out_embedder=BaiscEmbedder(config["symbol_size"],config["embedding_size"],config["dropout_ratio"])

        self.encoder = BasicRNNEncoder(config["embedding_size"], config['hidden_size'], config['num_encoder_layers'], config['rnn_cell_type'],
                                       config['dropout_ratio'], config["bidirectional"])
        if self.attention:
            self.decoder=AttentionalRNNDecoder(config["embedding_size"]+config["latent_size"],config["hidden_size"],config["hidden_size"],\
                                                config["num_decoder_layers"],config["rnn_cell_type"],config["dropout_ratio"])
        else:
            self.decoder = BasicRNNDecoder(config["embedding_size"]+config["latent_size"], config['hidden_size'], config['num_decoder_layers'], config['rnn_cell_type'],
                                       config['dropout_ratio'])

        self.dropout = nn.Dropout(config['dropout_ratio'])
        self.out = nn.Linear(config['hidden_size'], config["symbol_size"])

        if self.rnn_cell_type == "lstm":
            self.hidden_to_mean = nn.Linear(self.num_directions * config['hidden_size'], config['latent_size'])
            self.hidden_to_logvar = nn.Linear(self.num_directions * config['hidden_size'], config['latent_size'])
            self.latent_to_hidden = nn.Linear(config['latent_size'], 2 * config['hidden_size'])
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            self.hidden_to_mean = nn.Linear(self.num_directions * config['hidden_size'], config['latent_size'])
            self.hidden_to_logvar = nn.Linear(self.num_directions * config['hidden_size'], config['latent_size'])
            self.latent_to_hidden = nn.Linear(config['latent_size'], 2 * config['hidden_size'])
        else:
            raise ValueError("No such rnn type {} for RNNVAE.".format(self.rnn_cell_type))

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
            token_logits=self.generate_t(decoder_inputs,hidden_states,z)
            return token_logits
        else:
            all_outputs=self.generate_without_t(decoder_inputs,hidden_states,z)
            return all_outputs
    
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
                token_logit = self.out(decoder_output)
                predict=torch.nn.functional.log_softmax(token_logit,dim=1)
                output=topk_sampling(predict)
                token_logits.append(predict)

                if self.share_vocab:
                    output=self.decode(output)
                    decoder_input=self.out_embedder(output)
                else:
                    decoder_input=self.out_embedder(output)
                decoder_input=torch.cat((decoder_input,z.unsqueeze(1).repeat(1,decoder_input.size(1),1)),dim=2)
            token_logits=torch.cat(token_logits,dim=1)
            token_logits=token_logits.view(-1,token_logits.size(-1))
        return token_logits
    
    def generate_without_t(self,encoder_outputs,decoder_input,decoder_hidden,z):
        all_outputs=[]
        for _ in range(self.max_length):
            decoder_input=torch.cat((decoder_input,z.unsqueeze(1).repeat(1,decoder_input.size(1),1)),dim=2)
            outputs, decoder_hidden = self.decoder(input_embeddings=decoder_input, hidden_states=decoder_hidden,encoder_outputs=encoder_outputs)
            token_logits = self.out(outputs)
            predict=torch.nn.functional.log_softmax(token_logits,dim=1)
            output = topk_sampling(predict)
            
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