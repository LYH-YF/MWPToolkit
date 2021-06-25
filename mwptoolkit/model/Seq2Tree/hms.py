import torch
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import HWCPEncoder
from mwptoolkit.module.Decoder.tree_decoder import HMSDecoder
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.utils.enum_type import SpecialTokens,NumMask

class HMS(nn.Module):
    def __init__(self,config,dataset):
        super(HMS,self).__init__()
        self.device=config["device"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio=config['dropout_ratio']
        self.beam_size = config['beam_size']
        self.output_length = config['max_output_len']
        self.share_vacab=config['share_vocab']

        self.span_size=dataset.max_span_size
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.operator_list = dataset.operator_list

        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.in_word2idx = dataset.in_word2idx
        self.in_idx2word = dataset.in_idx2word
        
        self.mask_list = NumMask.number
        self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
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
        embedder=BaiscEmbedder(self.vocab_size,self.embedding_size,self.dropout_ratio)
        # if self.share_vacab:
        #     self.out_embedder=self.in_embedder
        # else:
        #     self.out_embedder=BaiscEmbedder(self.symbol_size,self.embedding_size,self.dropout_ratio)
        self.encoder=HWCPEncoder(embedder,self.embedding_size,self.hidden_size,self.span_size,self.dropout_ratio)
        self.decoder=HMSDecoder(embedder,self.hidden_size,self.dropout_ratio,self.operator_list,self.in_word2idx,self.out_idx2symbol,self.device)
        
        weight = torch.ones(self.symbol_size).to(self.device)
        pad = self.out_pad_token
        #self.loss = NLLLoss(weight, pad)
        self.loss = nn.NLLLoss(weight,pad,reduction='sum')
    def forward(self, input_variable, input_lengths,span_num_pos,word_num_poses, span_length=None,tree=None,
                target_variable=None, max_length=None, beam_width=None):
        num_pos=(span_num_pos,word_num_poses)
        if beam_width != None:
            beam_width = self.beam_size
            max_length = self.output_length
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
    
    def calculate_loss(self,batch_data):
        input_variable=batch_data["spans"]
        input_lengths=batch_data["spans len"]
        span_num_pos=batch_data["span num pos"]
        word_num_poses=batch_data["word num poses"]
        span_length=batch_data["span nums"]
        tree=batch_data["deprel tree"]
        target_variable=batch_data["equation"]
        if self.share_vacab:
            target_variable=self.convert_in_idx_2_out_idx(target_variable)

        num_pos=(span_num_pos,word_num_poses)
        max_length=None
        beam_width=None
        encoder_outputs, encoder_hidden = self.encoder(
            input_var=input_variable,
            input_lengths=input_lengths,
            span_length=span_length,
            tree=tree
        )
        decoder_outputs, _, _ = self.decoder(
            targets=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            max_length=max_length,
            beam_width=beam_width
        )
        # outputs=torch.stack(decoder_outputs,dim=1)
        # outputs=outputs.view(-1,outputs.size(-1))
        # self.loss.reset()
        # self.loss.eval_batch(outputs,target_variable.view(-1))
        # self.loss.backward()
        batch_size = span_length.size(0)
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss(step_output.contiguous().view(batch_size, -1), target_variable[:, step].view(-1))
        
        total_target_length = (target_variable != self.out_pad_token).sum().item()
        loss = loss / total_target_length
        loss.backward()
        return loss.item()

    def model_test(self,batch_data):
        input_variable=batch_data["spans"]
        input_lengths=batch_data["spans len"]
        span_num_pos=batch_data["span num pos"]
        word_num_poses=batch_data["word num poses"]
        span_length=batch_data["span nums"]
        tree=batch_data["deprel tree"]
        num_list = batch_data['num list']
        target_variable=batch_data["equation"]
        if self.share_vacab:
            target_variable=self.convert_in_idx_2_out_idx(target_variable)

        num_pos=(span_num_pos,word_num_poses)
        max_length=self.output_length
        beam_width=self.beam_size
        encoder_outputs, encoder_hidden = self.encoder(
            input_var=input_variable,
            input_lengths=input_lengths,
            span_length=span_length,
            tree=tree
        )
        _, _, sequence_symbols = self.decoder(
            targets=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            max_length=max_length,
            beam_width=beam_width
        )
        targets=self.convert_idx2symbol(target_variable,num_list)
        outputs=torch.cat(sequence_symbols,dim=1)
        outputs=self.convert_idx2symbol(outputs,num_list)
        return outputs,targets

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
