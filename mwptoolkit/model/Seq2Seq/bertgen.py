import random
from torch import nn
import torch
from transformers import BertModel, BertTokenizer
from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder

from mwptoolkit.module.Encoder.transformer_encoder import TransformerEncoder
from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder
from mwptoolkit.module.Embedder.position_embedder import PositionEmbedder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Attention.self_attention import SelfAttentionMask
from mwptoolkit.module.Strategy.beam_search import Beam_Search_Hypothesis
from mwptoolkit.module.Strategy.sampling import topk_sampling
from mwptoolkit.module.Strategy.greedy import greedy_search



class BERTGen(nn.Module):

    def __init__(self, config):
        super(BERTGen, self).__init__()
        self.device = config["device"]
        self.pretrained_model_path = config['pretrained_model_path']

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)

        self.eos_token_id = self.tokenizer.sep_token_id
        self.eos_token = self.tokenizer.sep_token

        self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        self.model = TransformerDecoder(config["embedding_size"],config["ffn_size"],config["num_decoder_layers"],\
                                            config["num_heads"],config["attn_dropout_ratio"],\
                                            config["attn_weight_dropout_ratio"],config["ffn_dropout_ratio"])

        self.out_symbol2idx = config["out_symbol2idx"]
        self.out_idx2symbol = config["out_idx2symbol"]

        self.max_output_len = config["max_output_len"]
        self.share_vocab = config["share_vocab"]
        self.decoding_strategy = config["decoding_strategy"]
        self.teacher_force_ratio = 0.9

        print (config["out_symbol2idx"], config["out_sos_token"])
        self.out_pad_idx = config["out_symbol2idx"]["<PAD>"]
        self.out_sos_idx = config["out_symbol2idx"]["<SOS>"]
        self.out_eos_idx = config["out_symbol2idx"]["<EOS>"]
        self.out_unk_idx = config["out_symbol2idx"]["<UNK>"]

        # print("config:", config)

        self.in_embedder = BaiscEmbedder(config["vocab_size"], config["embedding_size"],
                                         config["embedding_dropout_ratio"])

        self.out_embedder = BaiscEmbedder(config["symbol_size"], config["embedding_size"],
                                              config["embedding_dropout_ratio"])

        # self.pos_embedder = PositionEmbedder(config["embedding_size"], config["device"],
        #                                      config["embedding_dropout_ratio"], config["max_len"])
        self.pos_embedder = PositionEmbedder(config["embedding_size"], config["max_len"])
        self.self_attentioner = SelfAttentionMask()

        self.decoder = TransformerDecoder(config["embedding_size"], config["ffn_size"], config["num_decoder_layers"], \
                                          config["num_heads"], config["attn_dropout_ratio"], \
                                          config["attn_weight_dropout_ratio"], config["ffn_dropout_ratio"])
        self.out = nn.Linear(config["embedding_size"], config["symbol_size"])



    def forward(self, seq, target=None):

        srcs = []
        for idx, s in enumerate(seq):
            src = self.tokenizer.encode(seq[idx])
            srcs.append(src)
        src_length = max([len(_) for _ in srcs])
        for i in range(len(srcs)):
            srcs[i] =  [self.tokenizer.cls_token_id] + srcs[i] + (src_length - len(srcs[i])) * [self.tokenizer.pad_token_id]
        src_length = src_length + 1
        srcs_tensor = torch.LongTensor(srcs).to(self.device)
        src_feat = self.encoder(srcs_tensor)[0]  # src_feat: torch.Size([4, 70, 768])

        source_padding_mask = torch.eq(srcs_tensor, self.tokenizer.pad_token_id)

        if target != None:

            tgts = []
            for idx, t in enumerate(target):
                tgt = []
                for _ in t:
                    if _ not in self.out_symbol2idx:
                        # print (self.out_symbol2idx)
                        tgt.append(self.out_symbol2idx['<UNK>'])
                    else:
                        tgt.append(self.out_symbol2idx[_] )
                # tgt = [self.out_symbol2idx[_] for _ in t]
                tgts.append(tgt)

            target_length = max([len(_) for _ in tgts])
            for i in range(len(tgts)):
                tgts[i] = tgts[i] + [self.out_eos_idx] + (target_length - len(tgts[i])) * [
                    self.out_pad_idx]

            tgts = torch.LongTensor(tgts).to(self.device)

            token_logits = self.generate_t(src_feat, tgts, source_padding_mask)
            return token_logits, tgts
        else:
            all_output = self.generate_without_t(src_feat, source_padding_mask)
            return all_output, None

    def generate_t(self, encoder_outputs, target, source_padding_mask):
        with_t = random.random()
        seq_len = target.size(1)
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]

            decoder_inputs = self.pos_embedder(self.out_embedder(target))
            self_padding_mask = torch.eq(target, self.out_pad_idx)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            decoder_outputs = self.decoder(decoder_inputs,
                                           self_padding_mask=self_padding_mask,
                                           self_attn_mask=self_attn_mask,
                                           external_states=encoder_outputs,
                                           external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        else:
            token_logits = []
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_outputs = self.decoder(decoder_input,
                                               self_attn_mask=self_attn_mask,
                                               external_states=encoder_outputs,
                                               external_padding_mask=source_padding_mask)

                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                if self.decoding_strategy == "topk_sampling":
                    output = topk_sampling(token_logit, top_k=5)
                elif self.decoding_strategy == "greedy_search":
                    output = greedy_search(token_logit)
                else:
                    raise NotImplementedError
                if self.share_vocab:
                    pre_tokens.append(self.decode(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def generate_without_t(self, encoder_outputs, source_padding_mask):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1).to(device)
        pre_tokens = [input_seq]
        all_outputs = []
        for gen_idx in range(self.max_output_len):
            self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
            # decoder_input = self.out_embedder(input_seq) + self.pos_embedder(input_seq)
            decoder_input = self.pos_embedder(self.out_embedder(input_seq))
            decoder_outputs = self.decoder(decoder_input,
                                           self_attn_mask=self_attn_mask,
                                           external_states=encoder_outputs,
                                           external_padding_mask=source_padding_mask)

            token_logits = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
            if self.decoding_strategy == "topk_sampling":
                output = topk_sampling(token_logits, top_k=5)
            elif self.decoding_strategy == "greedy_search":
                output = greedy_search(token_logits)
            else:
                raise NotImplementedError
            all_outputs.append(output)
            if self.share_vocab:
                pre_tokens.append(self.decode(output))
            else:
                pre_tokens.append(output)
            input_seq = torch.cat(pre_tokens, dim=1)
        all_outputs = torch.cat(all_outputs, dim=1)
        # print (all_outputs)
        all_outputs = self.decode_(all_outputs)
        # print ("2", all_outputs)
        return all_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = self.tokenizer.decode(outputs[b])
            symbols = self.tokenizer.tokenize(symbols)
            symbols = [self.out_idx2symbol[_] for _ in outputs[b]]
            symbols_ = []
            for token in symbols:
                # if '/' == token[0] and len(token) == 2 and (
                #         '+' == token[1] or '-' == token[1] or '*' == token[1] or '/' == token[1]):
                #     symbols_.append(token[0])
                #     symbols_.append(token[1:])
                if token =="<EOS>":
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            # print ("symbols",symbols)
            all_outputs.append(symbols)
        # print (all_outputs)
        return all_outputs

    def decode(self, output):
        device = output.device

        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return output

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters