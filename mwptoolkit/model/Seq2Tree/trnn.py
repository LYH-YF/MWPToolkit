import copy
import random

from mwptoolkit.module.Decoder.rnn_decoder import AttentionalRNNDecoder
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.loss.cross_entropy_loss import CrossEntropyLoss
import torch
from torch import nn

#from mwptoolkit.module.Layer.tree_layers import Node,BinaryTree
from mwptoolkit.module.Layer.tree_layers import RecursiveNN
from mwptoolkit.module.Encoder.rnn_encoder import SelfAttentionRNNEncoder, BasicRNNEncoder
from mwptoolkit.module.Attention.seq_attention import SeqAttention
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec
from mwptoolkit.utils.data_structure import Node, BinaryTree
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class TRNN(nn.Module):
    def __init__(self, config, dataset):
        super(TRNN,self).__init__()
        self.embedding_size=config["embedding_size"]
        self.dropout_ratio = config['dropout_ratio']
        self.bidirectional = config["bidirectional"]
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.decode_hidden_size = config["decode_hidden_size"]
        self.attention = True
        self.num_layers = config["num_layers"]
        self.share_vocab = config["share_vocab"]
        self.teacher_force_ratio = config["teacher_force_ratio"]
        self.encoder_rnn_cell_type = config["encoder_rnn_cell_type"]
        self.decoder_rnn_cell_type = config["decoder_rnn_cell_type"]
        self.max_gen_len=config["max_output_len"]
        
        self.mask_list = NumMask.number
        self.in_idx2word = dataset.in_idx2word
        self.out_idx2symbol=dataset.out_idx2symbol
        self.temp_idx2symbol=dataset.temp_idx2symbol
        self.vocab_size=len(dataset.in_idx2word)
        self.symbol_size=len(dataset.out_idx2symbol)
        self.temp_symbol_size=len(dataset.temp_idx2symbol)
        self.operator_nums=len(dataset.operator_list)
        self.operator_list=dataset.operator_list
        self.generate_list=[SpecialTokens.UNK_TOKEN]+dataset.generate_list
        self.generate_idx = [self.in_idx2word.index(num) for num in self.generate_list]
        
        if self.share_vocab:
            self.sos_token_idx = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.sos_token_idx = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        try:
            self.out_sos_token = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        
        try:
            self.temp_sos_token = dataset.temp_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.temp_sos_token = None
        try:
            self.temp_eos_token = dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.temp_eos_token = None
        try:
            self.temp_pad_token = dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.temp_pad_token = None

        # seq2seq module
        self.seq2seq_in_embedder=BaiscEmbedder(self.vocab_size,self.embedding_size,self.dropout_ratio)
        if self.share_vocab:
            self.seq2seq_out_embedder=self.seq2seq_in_embedder
        else:
            self.seq2seq_out_embedder=BaiscEmbedder(self.temp_symbol_size,self.embedding_size,self.dropout_ratio)
        self.seq2seq_encoder=BasicRNNEncoder(self.embedding_size,self.hidden_size,self.num_layers,\
                                                self.encoder_rnn_cell_type,self.dropout_ratio,self.bidirectional)
        self.seq2seq_decoder=AttentionalRNNDecoder(self.embedding_size,self.decode_hidden_size,self.hidden_size,\
                                                    self.num_layers,self.decoder_rnn_cell_type,self.dropout_ratio)
        self.seq2seq_gen_linear = nn.Linear(self.hidden_size, self.temp_symbol_size)
        #answer module
        self.answer_in_embedder = BaiscEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.answer_encoder=SelfAttentionRNNEncoder(self.embedding_size,self.hidden_size,self.embedding_size,self.num_layers,\
                                                    self.encoder_rnn_cell_type,self.dropout_ratio,self.bidirectional)
        self.answer_rnn = RecursiveNN(self.embedding_size, self.operator_nums, self.operator_list)

        weight = torch.ones(self.temp_symbol_size).to(config["device"])
        pad = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.seq2seq_loss = NLLLoss(weight, pad)
        weight2=torch.ones(self.operator_nums).to(config["device"])
        self.ans_module_loss=CrossEntropyLoss(weight2,size_average=True)

        self.wrong=0

    def calculate_loss(self,batch_data):
        # first stage:train seq2seq
        seq2seq_loss=self.seq2seq_calculate_loss(batch_data)

        # second stage: train answer module
        answer_loss=self.ans_module_calculate_loss(batch_data)

        return seq2seq_loss,answer_loss
    
    def model_test(self,batch_data):
        seq = batch_data['question']
        seq_length = batch_data['ques len']
        target = batch_data['equation']
        num_pos = batch_data['num pos']
        num_list = batch_data["num list"]

        batch_size = seq.size(0)
        device = seq.device

        seq_emb = self.seq2seq_in_embedder(seq)
        encoder_outputs, encoder_hidden = self.seq2seq_encoder(seq_emb, seq_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.encoder_rnn_cell_type == 'lstm'):
                encoder_hidden = (encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous())
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder_rnn_cell_type == self.decoder_rnn_cell_type:
            pass
        elif (self.encoder_rnn_cell_type == 'gru') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'rnn') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'lstm') and (self.decoder_rnn_cell_type == 'gru' or self.decoder_rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass

        decoder_inputs = self.init_seq2seq_decoder_inputs(None, device, batch_size)
        output_template = self.seq2seq_generate_without_t(encoder_outputs, encoder_hidden, decoder_inputs)
        template = self.convert_temp_idx2symbol(output_template)
        
        device = seq.device
        seq_emb = self.answer_in_embedder(seq)
        encoder_output, encoder_hidden = self.answer_encoder(seq_emb, seq_length)

        batch_size = encoder_output.size(0)
        generate_num = torch.tensor(self.generate_idx).to(device)
        generate_emb = self.answer_in_embedder(generate_num)
        equations = []
        for b_i in range(batch_size):
            try:
                tree_i = self.template2tree(template[b_i])
            except:
                equations.append([])
                continue
            look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            nodes_pred = self.answer_rnn.test(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
            tree_i.root = nodes_pred
            equation = self.tree2equation(tree_i)
            #equation = self.symbol2idx(equation)
            equations.append(equation)
            # tree_i=self.template2tree(template[b_i])
            # equations.append([])
        equations=self.mask2num(equations,num_list)
        targets=self.convert_idx2symbol(target,num_list)
        temp_t=self.convert_temp_idx2symbol(batch_data['template'])
        return equations,targets,template,temp_t
    
    def seq2seq_calculate_loss(self, batch_data):
        r"""calculate loss of a batch data.
        """
        seq = batch_data['question']
        seq_length = batch_data['ques len']
        target = batch_data['template']

        batch_size = seq.size(0)
        device = seq.device

        seq_emb = self.seq2seq_in_embedder(seq)
        encoder_outputs, encoder_hidden = self.seq2seq_encoder(seq_emb, seq_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if (self.encoder_rnn_cell_type == 'lstm'):
                encoder_hidden = (encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous())
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder_rnn_cell_type == self.decoder_rnn_cell_type:
            pass
        elif (self.encoder_rnn_cell_type == 'gru') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'rnn') and (self.decoder_rnn_cell_type == 'lstm'):
            encoder_hidden = (encoder_hidden, encoder_hidden)
        elif (self.encoder_rnn_cell_type == 'lstm') and (self.decoder_rnn_cell_type == 'gru' or self.decoder_rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass

        decoder_inputs = self.init_seq2seq_decoder_inputs(target, device, batch_size)
        token_logits = self.seq2seq_generate_t(encoder_outputs, encoder_hidden, decoder_inputs)
        if self.share_vocab:
            target = self.convert_in_idx_2_temp_idx(target)
        self.seq2seq_loss.reset()
        self.seq2seq_loss.eval_batch(token_logits, target.view(-1))
        self.seq2seq_loss.backward()
        return self.seq2seq_loss.get_loss()
    
    def ans_module_calculate_loss(self,batch_data):
        seq=batch_data["question"]
        seq_length=batch_data["ques len"]
        num_pos=batch_data["num pos"]
        for idx,equ in enumerate(batch_data["equ_source"]):
            batch_data["equ_source"][idx]=equ.split(" ")
        template=batch_data["equ_source"]

        device = seq.device
        seq_emb = self.answer_in_embedder(seq)
        encoder_output, encoder_hidden = self.answer_encoder(seq_emb, seq_length)
        batch_size = encoder_output.size(0)
        generate_num = torch.tensor(self.generate_idx).to(device)
        generate_emb = self.answer_in_embedder(generate_num)

        batch_prob = []
        batch_target = []
        for b_i in range(batch_size):
            try:
                tree_i = self.template2tree(template[b_i])
            except IndexError:
                self.wrong+=1
                continue
            look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            assert len(look_up)==len(num_embedding)
            #tree_i=tree[b_i]
            prob, target = self.answer_rnn(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
            if prob != []:
                batch_prob.append(prob)
                batch_target.append(target)
        
        self.ans_module_loss.reset()
        for b_i in range(len(batch_target)):
            #output=torch.nn.functional.log_softmax(batch_prob[b_i],dim=1)
            self.ans_module_loss.eval_batch(batch_prob[b_i], batch_target[b_i].view(-1))
        self.ans_module_loss.backward()
        return self.ans_module_loss.get_loss()
    
    def seq2seq_generate_t(self, encoder_outputs, encoder_hidden, decoder_inputs):
        with_t = random.random()
        if with_t < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden)
            token_logits = self.seq2seq_gen_linear(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
            token_logits = torch.nn.functional.log_softmax(token_logits, dim=1)
            #token_logits=torch.log_softmax(token_logits,dim=1)
        else:
            seq_len = decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            token_logits = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
                #attn_list.append(attn)
                step_output = decoder_output.squeeze(1)
                token_logit = self.seq2seq_gen_linear(step_output)
                predict = torch.nn.functional.log_softmax(token_logit, dim=1)
                #predict=torch.log_softmax(token_logit,dim=1)
                output = predict.topk(1, dim=1)[1]
                token_logits.append(predict)

                if self.share_vocab:
                    output = self.convert_temp_idx_2_in_idx(output)
                    decoder_input = self.seq2seq_out_embedder(output)
                else:
                    decoder_input = self.seq2seq_out_embedder(output)
            token_logits = torch.stack(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def seq2seq_generate_without_t(self, encoder_outputs, encoder_hidden, decoder_input):
        all_outputs = []
        decoder_hidden = encoder_hidden
        for idx in range(self.max_gen_len):
            if self.attention:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            token_logits = self.seq2seq_gen_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logits, dim=1)
            output = predict.topk(1, dim=1)[1]

            all_outputs.append(output)
            if self.share_vocab:
                output = self.convert_temp_idx_2_in_idx(output)
                decoder_input = self.seq2seq_out_embedder(output)
            else:
                decoder_input = self.seq2seq_out_embedder(output)
        all_outputs = torch.cat(all_outputs, dim=1)
        return all_outputs

    def template2tree(self, template):
        tree = BinaryTree()
        tree.equ2tree_(template)
        return tree

    def tree2equation(self, tree):
        equation = tree.tree2equ(tree.root)
        return equation
    
    def init_seq2seq_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).to(device).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.seq2seq_out_embedder(decoder_inputs)
        return decoder_inputs

    def convert_temp_idx_2_in_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.temp_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_temp_idx(self, output):
        device = output.device

        batch_size = output.size(0)
        seq_len = output.size(1)

        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.temp_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return decoded_output

    def convert_temp_idx2symbol(self,output):
        batch_size = output.size(0)
        seq_len = output.size(1)
        symbol_list = []
        for b_i in range(batch_size):
            symbols = []
            for s_i in range(seq_len):
                idx=output[b_i][s_i]
                if idx in [self.temp_sos_token, self.temp_eos_token, self.temp_pad_token]:
                    break
                symbol = self.temp_idx2symbol[idx]
                symbols.append(symbol)
            symbol_list.append(symbols)
        return symbol_list

    def convert_idx2symbol(self,output,num_list):
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

    def symbol2idx(self, symbols):
        r"""symbol to idx
        equation symbol to equation idx
        """
        outputs = []
        for symbol in symbols:
            if symbol not in self.out_idx2symbol:
                idx = self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)
            else:
                idx = self.out_idx2symbol.index(symbol)
            outputs.append(idx)
        return outputs

    def mask2num(self,output,num_list):
        batch_size=len(output)
        output_list = []
        for b_i in range(batch_size):
            res=[]
            seq_len=len(output[b_i])
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                symbol = output[b_i][s_i]
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
        


# class TRNN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         temp_config = copy.deepcopy(config)
#         self.generate_list = config["generate_list"]
#         self.temp_idx2symbol = config["temp_idx2symbol"]
#         self.out_idx2symbol = config["out_idx2symbol"]
#         self.in_idx2word = config["in_idx2word"]
#         self.bidirectional = config["bidirectional"]
#         self.hidden_size = config["hidden_size"]
#         temp_config["out_idx2symbol"] = temp_config["temp_idx2symbol"]
#         temp_config["out_symbol2idx"] = temp_config["temp_symbol2idx"]
#         temp_config["symbol_size"] = temp_config["temp_symbol_size"]
#         self.seq2seq = RNNEncDec(temp_config)
#         self.embedder = BaiscEmbedder(temp_config["vocab_size"], temp_config["embedding_size"], temp_config["dropout_ratio"])
#         self.attn_encoder=SelfAttentionRNNEncoder(temp_config["embedding_size"],temp_config["hidden_size"],temp_config["embedding_size"],temp_config["num_layers"],\
#                                                     temp_config["rnn_cell_type"],temp_config["dropout_ratio"],temp_config["bidirectional"])
#         self.recursivenn = RecursiveNN(temp_config["embedding_size"], temp_config["operator_nums"], temp_config["operator_list"])

#         weight2=torch.ones(temp_config["operator_nums"]).to(self.config["device"])
#         self.ans_module_loss=NLLLoss(weight2)

#     def forward(self, seq, seq_length, num_pos, target=None):
#         if target != None:
#             return self.generate_t()
#         else:
#             templates, equations = self.generate_without_t(seq, seq_length, num_pos)
#             return templates, equations
#     def calculate_loss_seq2seq(self,batch_data):
#         #batch_data["question"], batch_data["ques len"], batch_data["template"]
#         new_batch_data={}
#         new_batch_data["question"]=batch_data["question"]
#         new_batch_data["ques len"]=batch_data['ques len']
#         new_batch_data['equation']=batch_data['template']
#         return self.seq2seq.calculate_loss(new_batch_data)
#     def calculate_loss_ans_module(self,batch_data):
#         seq=batch_data["question"]
#         seq_length=batch_data["ques len"]
#         num_pos=batch_data["num pos"]
#         template=batch_data["equ_source"]

#         device = seq.device
#         seq_emb = self.embedder(seq)
#         encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)
#         batch_size = encoder_output.size(0)
#         generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
#         generate_num = torch.tensor(generate_num).to(device)
#         generate_emb = self.embedder(generate_num)

#         batch_prob = []
#         batch_target = []
#         for b_i in range(batch_size):
#             look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
#             num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
#             #tree_i=tree[b_i]
#             try:
#                 tree_i = self.template2tree(template[b_i])
#             except IndexError:
#                 continue
#             prob, target = self.recursivenn.forward(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
#             if prob != []:
#                 batch_prob.append(prob)
#                 batch_target.append(target)
        
#         self.ans_module_loss.reset()
#         for b_i in range(len(target)):
#             output=torch.nn.functional.log_softmax(batch_prob[b_i],dim=1)
#             self.ans_module_loss.eval_batch(output, target[b_i].view(-1))
#         self.ans_module_loss.backward()
#         return self.ans_module_loss.get_loss()
#     def model_test(self,batch_data):
#         new_batch_data={}
#         new_batch_data["question"]=batch_data["question"]
#         new_batch_data["ques len"]=batch_data['ques len']
#         new_batch_data['equation']=batch_data['template']
#         outputs,_ = self.seq2seq.model_test(batch_data)
#         templates = self.idx2symbol(outputs)
#     def generate_t(self):
#         raise NotImplementedError("use model.seq2seq_forward() to train seq2seq module, use model.answer_module_forward() to train answer module.")

#     def generate_without_t(self, seq, seq_length, num_pos):
#         outputs = self.seq2seq(seq, seq_length)
#         templates = self.idx2symbol(outputs)
#         equations = self.test_ans_module(seq, seq_length, num_pos, templates)
#         return outputs, equations

#     def seq2seq_forward(self, seq, seq_length, target=None):
#         return self.seq2seq(seq, seq_length, target)

#     def answer_module_forward(self, seq, seq_length, num_pos, template):
#         seq, seq_length, num_pos, template=0
#         device = seq.device
#         seq_emb = self.embedder(seq)
#         encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)
#         batch_size = encoder_output.size(0)
#         generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
#         generate_num = torch.tensor(generate_num).to(device)
#         generate_emb = self.embedder(generate_num)
#         # tree=[]
#         # for temp in template:
#         #     tree.append(self.template2tree(temp))
#         batch_prob = []
#         batch_target = []
#         for b_i in range(batch_size):
#             look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
#             num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
#             #tree_i=tree[b_i]
#             try:
#                 tree_i = self.template2tree(template[b_i])
#             except IndexError:
#                 continue
#             prob, target = self.recursivenn.forward(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
#             if prob != []:
#                 batch_prob.append(prob)
#                 batch_target.append(target)
#         return batch_prob, batch_target

#     def test_ans_module(self, seq, seq_length, num_pos, template):
#         self.wrong=0
#         device = seq.device
#         seq_emb = self.embedder(seq)
#         encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)

#         batch_size = encoder_output.size(0)
#         generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
#         generate_num = torch.tensor(generate_num).to(device)
#         generate_emb = self.embedder(generate_num)
#         tree = []
#         equations = []
#         # for temp in template:
#         #     tree.append(self.template2tree(temp))
#         for b_i in range(batch_size):
#             look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
#             num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
#             try:
#                 tree_i = self.template2tree(template[b_i])
#             except:
#                 equations.append([])
#                 self.wrong+=1
#                 continue
#             nodes_pred = self.recursivenn.test(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
#             tree_i.root = nodes_pred
#             equation = self.tree2equation(tree_i)
#             equation = self.symbol2idx(equation)
#             equations.append(equation)
#             # tree_i=self.template2tree(template[b_i])
#             # equations.append([])
#         return equations

#     def template2tree(self, template):
#         tree = BinaryTree()
#         tree.equ2tree_(template)
#         return tree

#     def tree2equation(self, tree):
#         equation = tree.tree2equ(tree.root)
#         return equation

#     def idx2symbol(self, output):
#         r"""idx to symbol
#         tempalte idx to template symbol
#         """
#         batch_size = output.size(0)
#         seq_len = output.size(1)
#         symbol_list = []
#         for b in range(batch_size):
#             symbols = []
#             for i in range(seq_len):
#                 symbols.append(self.temp_idx2symbol[output[b, i]])
#             symbol_list.append(symbols)
#         return symbol_list

#     def symbol2idx(self, symbols):
#         r"""symbol to idx
#         equation symbol to equation idx
#         """
#         outputs = []
#         for symbol in symbols:
#             if symbol not in self.out_idx2symbol:
#                 idx = self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)
#             else:
#                 idx = self.out_idx2symbol.index(symbol)
#             outputs.append(idx)
#         return outputs
