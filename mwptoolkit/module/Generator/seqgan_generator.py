import random
import math
import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder
from mwptoolkit.module.Decoder.rnn_decoder import AttentionalRNNDecoder, BasicRNNDecoder
class SeqGANGenerator(nn.Module):
    r"""The generator of SeqGAN.
    """
    def __init__(self, config):
        super(SeqGANGenerator, self).__init__()
        self.bidirectional=config["bidirectional"]
        self.rnn_cell_type=config["rnn_cell_type"]
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_gen_len=30
        self.monte_carlo_num = config['Monte_Carlo_num']
        self.eval_generate_num = config['eval_generate_num']
        self.share_vocab= config["share vocab"]
        self.teacher_force_ratio=0.9
        self.batch_size=64
        self.device=config["device"]
        self.num_layers=config["num_layers"]
        
        self.out_sos_token = config["out_sos_token"]
        self.out_eos_token=config["out_eos_token"]
        self.out_pad_token=config["out_pad_token"]
        self.vocab_size = config["vocab_size"]

        self.in_embedder=BasicEmbedder(config["vocab_size"],config["generator_embedding_size"],config["dropout_ratio"])
        if config["share_vocab"]:
            self.out_embedder=self.in_embedder
        else:
            self.out_embedder=BasicEmbedder(config["symbol_size"],config["generator_embedding_size"],config["dropout_ratio"])
        
        self.encoder=BasicRNNEncoder(config["generator_embedding_size"],config["hidden_size"],config["num_layers"],\
                                        config["rnn_cell_type"],config["dropout_ratio"])
        # self.decoder=AttentionalRNNDecoder(config["generator_embedding_size"],config["hidden_size"],config["hidden_size"],\
        #                                     config["num_layers"],config["rnn_cell_type"],config["dropout_ratio"])
        self.decoder=BasicRNNDecoder(config["generator_embedding_size"],config["hidden_size"],config["num_layers"],\
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
            all_output,token_logits,outputs,P=self.generate_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return all_output,token_logits,outputs,P
        else:
            all_outputs=self.generate_without_t(encoder_outputs,encoder_hidden,decoder_inputs)
            return all_outputs,None,None,None
    
    def generate_t(self,encoder_outputs,encoder_hidden,decoder_inputs):
        batch_size=encoder_outputs.size(0)
        fake_samples = self.sample(batch_size)
        with_t=random.random()
        seq_len=decoder_inputs.size(1)
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:,0].unsqueeze(1)
        monte_carlo_outputs=[]
        token_logits=[]
        P=[]
        all_output=[]
        for idx in range(seq_len):
            if with_t<self.teacher_force_ratio:
                tokens = decoder_inputs[:,idx].unsqueeze(1)
            decoder_input=self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict=torch.nn.functional.log_softmax(token_logit,dim=1)

            tokens = fake_samples[ : , idx].unsqueeze(1) # b
            if self.share_vocab:
                tokens=self.decode(tokens)
            P_t = torch.gather(predict, 1, tokens).squeeze(1) # b
            monte_carlo_output=self.Monte_Carlo_search(tokens,decoder_hidden,fake_samples,idx,seq_len)

            monte_carlo_outputs.append(monte_carlo_output)
            P.append(P_t)
            all_output.append(tokens)
            token_logits.append(predict)
        all_output=torch.cat(all_output,dim=1)
        token_logits=torch.stack(token_logits,dim=1)
        token_logits=token_logits.view(-1,token_logits.size(-1))
        return all_output,token_logits,monte_carlo_outputs,P
    
    def generate_without_t(self,encoder_outputs,encoder_hidden,decoder_inputs):
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:,0].unsqueeze(1)
        all_output=[]
        for idx in range(self.max_gen_len):
            
            decoder_input=self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict=torch.nn.functional.log_softmax(token_logit,dim=1)

            tokens = predict.topk(1,dim=-1)[1]
            if self.share_vocab:
                tokens=self.decode(tokens)
            all_output.append(tokens)
        all_output=torch.cat(all_output,dim=1)
        return all_output
        
    def Monte_Carlo_search(self,tokens,decoder_hidden,fake_samples,idx,max_length):
        (h_prev, o_prev)=decoder_hidden
        self.eval()
        with torch.no_grad():
            monte_carlo_X = tokens.repeat_interleave(self.monte_carlo_num) # (b * M)
            monte_carlo_X = self.out_embedder(monte_carlo_X).unsqueeze(1) # 1 * (b * M) * e
            monte_carlo_h_prev = h_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim = 1) # 1 * (b * M) * h
            monte_carlo_o_prev = o_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim = 1) # 1 * (b * M) * h
            monte_carlo_output = torch.zeros(max_length, self.batch_size * self.monte_carlo_num, dtype = torch.long, device = self.device) # len * (b * M)

            for i in range(max_length - idx - 1):
                output, (monte_carlo_h_prev, monte_carlo_o_prev) = self.decoder(monte_carlo_X, (monte_carlo_h_prev, monte_carlo_o_prev))
                P = F.softmax(self.generate_linear(output), dim = -1).squeeze(0) # (b * M) * v
                for j in range(P.shape[0]):
                    monte_carlo_output[i + idx + 1][j] = torch.multinomial(P[j], 1)[0]
                monte_carlo_X = self.out_embedder(monte_carlo_output[i + idx + 1]).unsqueeze(1) # 1 * (b * M) * e

            monte_carlo_output = monte_carlo_output.permute(1, 0) # (b * M) * len
            monte_carlo_output[ : , : idx + 1] = fake_samples[ : , : idx + 1].repeat_interleave(self.monte_carlo_num, dim = 0)
        self.train()
        return monte_carlo_output
    
    def init_decoder_inputs(self,target,device,batch_size):
        pad_var = torch.LongTensor([self.out_sos_token]*batch_size).to(device).view(batch_size,1)
        if target != None:
            decoder_inputs=torch.cat((pad_var,target),dim=1)[:,:-1]
        else:
            decoder_inputs=pad_var
        return decoder_inputs
    
    def decode(self,output):
        device=output.device

        batch_size=output.size(0)
        decoded_output=[]
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output=torch.tensor(decoded_output).to(device).view(batch_size,-1)
        return output
    
    def _sample_batch(self,batch_size):
        r"""Sample a batch of generated sentence indice.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [batch_size, max_seq_length].
        """
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) # 1 * b * h
            o_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) # 1 * b * h
            prev_state = (h_prev, o_prev)
            tokens=self.init_decoder_inputs(None,self.device,batch_size)
            X = self.out_embedder(tokens) # 1 * b * e
            sentences = torch.zeros((batch_size, self.max_gen_len), dtype = torch.long).to(self.device)
            #sentences[0] = self.out_sos_token

            for i in range(0, self.max_gen_len):
                output, prev_state = self.decoder(X, prev_state)
                #output, prev_state = self.decoder(X)
                P = F.softmax(self.generate_linear(output), dim = -1).squeeze(0) # b * v
                for j in range(batch_size):
                    sentences[j][i] = torch.multinomial(P[j], 1)[0]
                X = self.out_embedder(sentences[:,i]).unsqueeze(1) # 1 * b * e

            #sentences = sentences.permute(1, 0) # b * l

            for i in range(batch_size):
                end_pos = (sentences[i] == self.out_eos_token).nonzero(as_tuple=False)
                if (end_pos.shape[0]):
                    sentences[i][end_pos[0][0] + 1 : ] = self.out_pad_token

        self.train()
        return sentences

    def sample(self, sample_num):
        r"""Sample sample_num generated sentence indice.

        Args:
            sample_num (int): The number to generate.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [sample_num, max_seq_length].
        """
        samples = []
        batch_num = math.ceil(sample_num / self.batch_size)
        for _ in range(batch_num):
            samples.append(self._sample_batch(sample_num))
        samples = torch.cat(samples, dim = 0)
        return samples[:sample_num, :]
    
    def pre_train(self,seq,seq_length,target):
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

        batch_size=encoder_outputs.size(0)
        with_t=random.random()
        seq_len=decoder_inputs.size(1)
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:,0].unsqueeze(1)
        monte_carlo_outputs=[]
        token_logits=[]
        P=[]
        all_output=[]
        for idx in range(seq_len):
            if with_t<self.teacher_force_ratio:
                tokens = decoder_inputs[:,idx].unsqueeze(1)
            decoder_input=self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict=torch.nn.functional.log_softmax(token_logit,dim=1)
            tokens=predict.topk(1,dim=1)[1]

            if self.share_vocab:
                tokens=self.decode(tokens)
            
            token_logits.append(predict)
        token_logits=torch.stack(token_logits,dim=1)
        token_logits=token_logits.view(-1,token_logits.size(-1))
        return token_logits



