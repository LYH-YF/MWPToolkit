# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/21 04:35:42
# @File: ept.py

import torch
from torch import nn
from transformers import AutoModel

from mwptoolkit.module.Encoder.transformer_encoder import TransformerEncoder
from mwptoolkit.module.Decoder.transformer_decoder import TransformerDecoder
from mwptoolkit.module.Decoder.ept_decoder import VanillaOpTransformer, ExpressionTransformer, ExpressionPointerTransformer
from mwptoolkit.module.Embedder.position_embedder import PositionEmbedder
from mwptoolkit.module.Embedder.basic_embedder import BasicEmbedder
from mwptoolkit.module.Attention.self_attention import SelfAttentionMask
from mwptoolkit.module.Strategy.beam_search import Beam_Search_Hypothesis
from mwptoolkit.module.Strategy.sampling import topk_sampling
from mwptoolkit.module.Strategy.greedy import greedy_search
from mwptoolkit.loss.smoothed_cross_entropy_loss import SmoothCrossEntropyLoss
from mwptoolkit.utils.enum_type import EPT as EPT_CON


def Submodule_types(decoder_type):
    if "vall" in decoder_type:
        return VanillaOpTransformer
    elif 'gen' in decoder_type:
        return ExpressionTransformer
    elif 'ptr' in decoder_type:
        return ExpressionPointerTransformer


class EPT(nn.Module):
    """
    Reference:
        Kim et al. "Point to the Expression: Solving Algebraic Word Problems using the Expression-Pointer Transformer Model" in EMNLP 2020.
    """
    def __init__(self, config, dataset):
        super(EPT, self).__init__()
        self.device = config["device"]
        self.max_output_len = config["max_output_len"]
        self.share_vocab = config["share_vocab"]
        self.decoding_strategy = config["decoding_strategy"]
        self.teacher_force_ratio = config["teacher_force_ratio"]
        self.task_type = config['task_type']

        try:
            self.in_pad_idx = dataset.in_word2idx["<pad>"]
        except:
            self.in_pad_idx = None
        self.in_word2idx = dataset.in_word2idx
        self.in_idx2word = dataset.in_idx2word
        self.mode = config["decoder"]
        if 'vall' in config["decoder"]:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol

            #self.out_pad_idx = self.in_pad_idx
            #self.out_sos_idx = config["in_word2idx"]["<SOS>"]
            self.decoder = VanillaOpTransformer(config, self.out_symbol2idx, self.out_idx2symbol)
        else:
            self.out_opsym2idx = dataset.out_opsym2idx
            self.out_idx2opsym = dataset.out_idx2opsymbol
            self.out_consym2idx = dataset.out_consym2idx
            self.out_idx2consym = dataset.out_idx2consymbol

            if 'gen' in config["decoder"]:
                self.decoder = ExpressionTransformer(config, self.out_opsym2idx, self.out_idx2opsym, self.out_consym2idx, self.out_idx2consym)
            elif 'ptr' in config["decoder"]:
                self.decoder = ExpressionPointerTransformer(config, self.out_opsym2idx, self.out_idx2opsym, self.out_consym2idx, self.out_idx2consym)
            #self.out_pad_idx = self.in_pad_idx
            #self.out_sos_idx = config["in_word2idx"]["<SOS>"]
        pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config[
            'transformers_pretrained_model']
        self.encoder = AutoModel.from_pretrained(pretrained_model_path)
        #self.encoder = TransformerEncoder(config["embedding_size"], config["ffn_size"], config["num_encoder_layers"], \
        #                                  config["num_heads"], config["attn_dropout_ratio"], \
        #                                  config["attn_weight_dropout_ratio"], config["ffn_dropout_ratio"])
        #self.decoder = TransformerDecoder(config["embedding_size"], config["ffn_size"], config["num_decoder_layers"], \
        #                                  config["num_heads"], config["attn_dropout_ratio"], \
        #                                  config["attn_weight_dropout_ratio"], config["ffn_dropout_ratio"])
        #self.decoder = Submodule_types(config["decoder"])(config)
        #self.out = nn.Linear(config["embedding_size"], config["symbol_size"])
        self.loss = SmoothCrossEntropyLoss()

    def forward(self, src, src_mask, num_pos, num_size, target=None, output_all_layers=False):
        """

        :param torch.Tensor src: input sequence.
        :param list src_mask: mask of input sequence.
        :param list num_pos: number position of input sequence.
        :param list num_size: number of numbers of input sequence.
        :param torch.Tensor target: target, default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        """
        encoder_output, encoder_layer_outputs = self.encoder_forward(src, src_mask, output_all_layers)
        max_numbers = max(num_size)
        if num_pos is not None:
            text_num, text_numpad = self.gather_vectors(encoder_output, num_pos, max_len=max_numbers)
        else:
            text_num = text_numpad = None

        token_logits, outputs, decoder_layer_outputs = self.decoder_forward(encoder_output, text_num, text_numpad, src_mask,
                                                                        target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)

        return token_logits, outputs, model_all_outputs

    def calculate_loss(self, batch_data:dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation','ques mask', 'num pos',
        'num size' and 'max numbers'.
        """
        src = torch.tensor(batch_data["question"]).to(self.device)
        src_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)
        num_pos = batch_data["num pos"]
        target = torch.tensor(batch_data["equation"]).to(self.device)
        num_size = batch_data["num size"]

        token_logits, _, all_layers = self.forward(src,src_mask,num_pos,num_size,target,output_all_layers=True)
        targets = all_layers['targets']

        self.loss.reset()

        for key, result in targets.items():

            predicted = token_logits[key].flatten(0, -2)
            result = self.shift_target(result)
            target = result.flatten()

            self.loss.eval_batch(predicted, target)
        self.loss.backward()
        batch_loss = self.loss.get_loss()
        return batch_loss

    def model_test(self, batch_data:dict) -> tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'equation','ques mask', 'num pos',
        'num size'.
        """
        src = torch.tensor(batch_data["question"]).to(self.device)
        src_mask = torch.BoolTensor(batch_data["ques mask"]).to(self.device)
        num_pos = batch_data["num pos"]
        num_size = batch_data["num size"]

        _, symbol_outputs, _ = self.forward(src, src_mask, num_pos, num_size)
        all_outputs = self.convert_idx2symbol(symbol_outputs, batch_data["num list"])
        targets = self.convert_idx2symbol(batch_data["equation"], batch_data["num list"])
        return all_outputs, targets

    def predict(self,batch_data:dict,output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        raise NotImplementedError

    def encoder_forward(self, src, src_mask, output_all_layers=False):
        encoder_outputs = self.encoder(input_ids=src, attention_mask=(~src_mask).float())
        encoder_output = encoder_outputs[0]

        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_output
            all_layer_outputs['inputs_representation'] = encoder_output
        return encoder_output, all_layer_outputs

    def decoder_forward(self, encoder_output, text_num, text_numpad, src_mask, target=None, output_all_layers=False):
        if target is not None:
            token_logits, targets = self.decoder(text=encoder_output, text_num=text_num, text_numpad=text_numpad,
                                                 text_pad=src_mask, equation=target)
            outputs = None
        else:
            max_len = self.max_output_len
            outputs, _ = self.decoder(text=encoder_output, text_num=text_num, text_numpad=text_numpad,
                                      text_pad=src_mask, beam=1, max_len=max_len)

            token_logits = None

            shape = list(outputs.shape)
            seq_len = shape[2]
            if seq_len < max_len:
                shape[2] = max_len
                tensor = torch.full(shape, fill_value=-1, dtype=torch.long)
                tensor[:, :, :seq_len] = outputs.cpu()
                outputs = tensor
            outputs.squeeze(1)
            targets = None
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['targets'] = targets

        return token_logits, outputs, all_layer_outputs

    def decode(self, output):
        device = output.device

        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).to(device).view(batch_size, -1)
        return output

    def gather_vectors(self, hidden: torch.Tensor, mask: torch.Tensor, max_len: int = 1):
        """
        Gather hidden states of indicated positions.

        :param torch.Tensor hidden:
            Float Tensor of hidden states.
            Shape [B, S, H], where B = batch size, S = length of sequence, and H = hidden dimension
        :param torch.Tensor mask:
            Long Tensor which indicates number indices that we're interested in. Shape [B, S].
        :param int max_len:
            Expected maximum length of vectors per batch. 1 by default.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :return:
            Tuple of Tensors:
            - [0]:  Float Tensor of indicated hidden states.
                    Shape [B, N, H], where N = max(number of interested positions, max_len)
            - [1]:  Bool Tensor of padded positions.
                    Shape [B, N].
        """
        # Compute the maximum number of indicated positions in the text
        max_len = max(mask.max().item(), max_len)
        batch_size, seq_len, hidden_size = hidden.shape

        # Storage for gathering hidden states
        gathered = torch.zeros(batch_size, max_len, hidden_size, dtype=hidden.dtype, device=hidden.device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=hidden.device)

        # Average hidden states for tokens representing a number
        for row in range(batch_size):
            for i in range(max_len):
                indices = (mask[row] == i).nonzero().view(-1).tolist()

                if len(indices) > 0:
                    begin = min(indices)
                    end = max(indices) + 1

                    # Copy masked positions. Take mean of number vectors.
                    gathered[row, i] = hidden[row, begin:end].mean(dim=0)
                    pad_mask[row, i] = False

        return gathered, pad_mask

    def shift_target(self, target: torch.Tensor, fill_value=-1) -> torch.Tensor:
        """
        Shift matrix to build generation targets.

        :param torch.Tensor target: Target tensor to build generation targets. Shape [B, T]
        :param fill_value: Value to be filled at the padded positions.
        :rtype: torch.Tensor
        :return: Tensor with shape [B, T], where (i, j)-entries are (i, j+1) entry of target tensor.
        """
        # Target does not require gradients.

        with torch.no_grad():
            pad_at_end = torch.full((target.shape[0], 1), fill_value=fill_value, dtype=target.dtype, device=target.device)
            return torch.cat([target[:, 1:], pad_at_end], dim=-1).contiguous()

    def convert_idx2symbol(self, output, num_list):
        #batch_size=output.size(0)
        '''batch_size=1'''
        output_list = []
        if "vall" in self.mode:
            for id, single in enumerate(output):
                output_list.append(self.out_expression_op(single, num_list[id]))
        else:
            for id, single in enumerate(output):
                output_list.append(self.out_expression_expr(single, num_list[id]))
        return output_list

    def out_expression_op(self, item, num_list):
        equation = []
        # Tokens after PAD_ID will be ignored.
        for i, token in enumerate(item.tolist()):
            if token != EPT_CON.PAD_ID:
                token = self.out_idx2sym[token]
                if token == EPT_CON.SEQ_NEW_EQN:
                    equation.clear()
                    continue
                elif token == EPT_CON.SEQ_END_EQN:
                    break
            else:
                break

            equation.append(token)

        return equation

    def out_expression_expr(self, item, num_list):
        expressions = []

        for token in item:
            # For each token in the item.
            # First index should be the operator.
            operator = self.out_idx2opsym[token[0]]
            if operator == EPT_CON.FUN_NEW_EQN:
                # If the operator is __NEW_EQN, we ignore the previously generated outputs.
                expressions.clear()
                continue

            if operator == EPT_CON.FUN_END_EQN:
                # If the operator is __END_EQN, we ignore the next outputs.
                break

            # Now, retrieve the operands
            operands = []
            for i in range(1, len(token), 2):
                # For each argument, we build two values: source and value.
                src = token[i]
                if src != EPT_CON.PAD_ID:
                    # If source is not a padding, compute the value.
                    src = EPT_CON.ARG_TOKENS[src]
                    operand = token[i + 1]
                    if src == EPT_CON.ARG_CON or "gen" in self.mode:
                        operand = self.out_idx2consym[operand]

                    if type(operand) is str and operand.startswith(EPT_CON.MEM_PREFIX):
                        operands.append((EPT_CON.ARG_MEM, int(operand[2:])))
                    else:
                        operands.append((src, operand))

            # Append an expression
            expressions.append((operator, operands))

        computation_history = []
        expression_used = []
        #print("expressions", expressions)
        for operator, operands in expressions:
            # For each expression.
            computation = []

            if operator == EPT_CON.FUN_NEW_VAR:
                # Generate new variable whenever __NEW_VAR() appears.
                computation.append(EPT_CON.FORMAT_VAR % len(computation_history))
            else:
                # Otherwise, form an expression tree
                for src, operand in operands:
                    # Find each operands from specified sources.
                    if src == EPT_CON.ARG_NUM and "ptr" in self.mode:
                        # If this is a number pointer, then replace it into number indices
                        computation.append(EPT_CON.FORMAT_NUM % operand)
                    elif src == EPT_CON.ARG_MEM:
                        # If this indicates the result of prior expression, then replace it with prior results

                        if operand < len(computation_history):
                            computation += computation_history[operand]
                            # Mark the prior expression as used.
                            expression_used[operand] = True
                        else:
                            # Expression is not found, then use UNK.
                            computation.append(EPT_CON.ARG_UNK)

                    else:
                        # Otherwise, this is a constant: append the operand itself.
                        computation.append(operand)

                # To make it as a postfix representation, append operator at the last.
                computation.append(operator)

            # Save current expression into the history.
            computation_history.append(computation)
            expression_used.append(False)

        # Find unused computation history. These are the top-level formula.
        computation_history = [equation for used, equation in zip(expression_used, computation_history) if not used]
        result = sum(computation_history, [])
        replace_result = []
        for word in result:
            if 'N_' in word:
                replace_result.append(str(num_list[int(word[2:])]['value']))
            elif 'C_' in word:
                replace_result.append(str(word[2:].replace('_', '.')))
            else:
                replace_result.append(word)
        if '=' in replace_result[:-1]:
            replace_result.append("<BRG>")
        return replace_result

    def __str__(self) -> str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters