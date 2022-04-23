import torch
from torch import nn

from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import HWCPEncoder
from mwptoolkit.module.Decoder.tree_decoder import HMSDecoder
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask


class HMS(nn.Module):
    def __init__(self, config, dataset):
        super(HMS, self).__init__()
        self.device = config["device"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config['dropout_ratio']
        self.beam_size = config['beam_size']
        self.output_length = config['max_output_len']
        self.share_vacab = config['share_vocab']

        self.span_size = dataset.max_span_size
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
        embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        embedder = self._init_embedding_params(dataset.trainset, dataset.in_idx2word, embedder)

        self.encoder = HWCPEncoder(embedder, self.embedding_size, self.hidden_size, self.span_size, self.dropout_ratio)
        self.decoder = HMSDecoder(embedder, self.hidden_size, self.dropout_ratio, self.operator_list, self.in_word2idx,
                                  self.out_idx2symbol, self.device)

        weight = torch.ones(self.symbol_size).to(self.device)
        pad = self.out_pad_token
        # self.loss = NLLLoss(weight, pad)
        self.loss = nn.NLLLoss(weight, pad, reduction='sum')

    def forward(self, input_variable, input_lengths, span_num_pos, word_num_poses, span_length=None, tree=None,
                target_variable=None, max_length=None, beam_width=None, output_all_layers=False):
        """

        :param input_variable:
        :param input_lengths:
        :param span_num_pos:
        :param word_num_poses:
        :param span_length:
        :param tree:
        :param target_variable:
        :param max_length:
        :param beam_width:
        :param output_all_layers:
        :return:
        """
        num_pos = (span_num_pos, word_num_poses)
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
        (token_logits, decoder_hidden, outputs) = output
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['encoder_outputs'] = encoder_outputs
            model_all_outputs['encoder_hidden'] = encoder_hidden
            model_all_outputs['decoder_hidden'] = decoder_hidden
            model_all_outputs['token_logits'] = token_logits
            model_all_outputs['outputs'] = outputs
        return token_logits, outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'spans', 'spans len', 'span num pos', 'word num poses',
        'span nums', 'deprel tree', 'num pos', 'equation'
        """
        input_variable = [torch.tensor(span_i_batch).to(self.device) for span_i_batch in batch_data["spans"]]
        input_lengths = torch.tensor(batch_data["spans len"]).long()
        span_num_pos = torch.LongTensor(batch_data["span num pos"]).to(self.device)
        word_num_poses = [torch.LongTensor(word_num_pos).to(self.device) for word_num_pos in
                          batch_data["word num poses"]]
        span_length = torch.tensor(batch_data["span nums"]).to(self.device)
        target_variable = torch.tensor(batch_data["equation"]).to(self.device)
        tree = batch_data["deprel tree"]
        if self.share_vacab:
            target_variable = self.convert_in_idx_2_out_idx(target_variable)

        num_pos = (span_num_pos, word_num_poses)
        max_length = None
        beam_width = None
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
        batch_size = span_length.size(0)
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss(step_output.contiguous().view(batch_size, -1), target_variable[:, step].view(-1))

        total_target_length = (target_variable != self.out_pad_token).sum().item()
        loss = loss / total_target_length
        loss.backward()
        return loss.item()

    def model_test(self, batch_data:dict) -> tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'spans', 'spans len', 'span num pos', 'word num poses',
        'span nums', 'deprel tree', 'num pos', 'equation', 'num list'
        """
        input_variable = [torch.tensor(span_i_batch).to(self.device) for span_i_batch in batch_data["spans"]]
        input_lengths = torch.tensor(batch_data["spans len"]).long()
        span_num_pos = torch.LongTensor(batch_data["span num pos"]).to(self.device)
        word_num_poses = [torch.LongTensor(word_num_pos).to(self.device) for word_num_pos in
                          batch_data["word num poses"]]
        span_length = torch.tensor(batch_data["span nums"]).to(self.device)
        target_variable = torch.tensor(batch_data["equation"]).to(self.device)
        tree = batch_data["deprel tree"]
        num_list = batch_data['num list']
        if self.share_vacab:
            target_variable = self.convert_in_idx_2_out_idx(target_variable)

        num_pos = (span_num_pos, word_num_poses)
        max_length = self.output_length
        beam_width = self.beam_size
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
        targets = self.convert_idx2symbol(target_variable, num_list)
        outputs = torch.cat(sequence_symbols, dim=1)
        outputs = self.convert_idx2symbol(outputs, num_list)
        return outputs, targets

    def predict(self, batch_data:dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        input_variable = [torch.tensor(span_i_batch).to(self.device) for span_i_batch in batch_data["spans"]]
        input_lengths = torch.tensor(batch_data["spans len"]).long()
        span_num_pos = torch.LongTensor(batch_data["span num pos"]).to(self.device)
        word_num_poses = [torch.LongTensor(word_num_pos).to(self.device) for word_num_pos in
                          batch_data["word num poses"]]
        span_length = torch.tensor(batch_data["span nums"]).to(self.device)
        tree = batch_data["deprel tree"]
        token_logits, symbol_outputs, model_all_layers = self.forward(input_variable, input_lengths, span_num_pos,
                                                                      word_num_poses, span_length, tree,
                                                                      output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_layers

    def _init_embedding_params(self, train_data, vocab, embedder):
        sentences = []
        for data in train_data:
            sentence = [SpecialTokens.SOS_TOKEN]
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentence += [SpecialTokens.EOS_TOKEN]
            sentences.append(sentence)
        embedder.init_embedding_params(sentences, vocab)

        return embedder

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
