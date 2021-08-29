# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 11:07:49
# @File: ept_decoder.py


from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


from mwptoolkit.module.Embedder.position_embedder import EPTPositionalEncoding
from mwptoolkit.module.Layer.transformer_layer import EPTTransformerLayer
from mwptoolkit.utils.enum_type import EPT
from mwptoolkit.module.Attention.multi_head_attention import EPTMultiHeadAttention, EPTMultiHeadAttentionWeights

class AveragePooling(nn.Module):
    """
    Layer class for computing mean of a sequence
    """

    def __init__(self, dim: int = -1, keepdim: bool = False):
        """
        Layer class for computing mean of a sequence

        :param int dim: Dimension to be averaged. -1 by default.
        :param bool keepdim: True if you want to keep averaged dimensions. False by default.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, tensor: torch.Tensor):# -> torch.Tensor:
        """
        Do average pooling over a sequence

        Args:
            tensor (torch.Tensor): FloatTensor to be averaged.
        
        Returns:
            torch.FloatTensor: Averaged result.
        """
        return tensor.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        # Extra representation for repr()
        return 'dim={dim}, keepdim={keepdim}'.format(**self.__dict__)


class Squeeze(nn.Module):
    """
    Layer class for squeezing a dimension
    """

    def __init__(self, dim: int = -1):
        """
        Layer class for squeezing a dimension

        :param int dim: Dimension to be squeezed, -1 by default.
        """
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor):# -> torch.Tensor:
        """
        Do squeezing

        Args:
            tensor (torch.Tensor): FloatTensor to be squeezed.
        
        Returns: 
            torch.FloatTensor: Squeezed result.
        """
        return tensor.squeeze(dim=self.dim)

    def extra_repr(self):
        # Extra representation for repr()
        return 'dim={dim}'.format(**self.__dict__)

def apply_module_dict(modules: nn.ModuleDict, encoded: torch.Tensor, **kwargs):# -> torch.Tensor:
    """
    Predict next entry using given module and equation.

    Args:
        modules (nn.ModuleDict): Dictionary of modules to be applied. Modules will be applied with ascending order of keys.
            We expect three types of modules: nn.Linear, nn.LayerNorm and MultiheadAttention.
        
        encoded (torch.Tensor): Float Tensor that represents encoded vectors. Shape [batch_size, equation_length, hidden_size].
        key_value (torch.Tensor): Float Tensor that represents key and value vectors when computing attention. Shape [batch_size, key_size, hidden_size].

        key_ignorance_mask (torch.Tensor):Bool Tensor whose True values at (b, k) make attention layer ignore k-th key on b-th item in the batch. Shape [batch_size, key_size].
        
        attention_mask (torch.BoolTensor): Bool Tensor whose True values at (t, k) make attention layer ignore k-th key when computing t-th query. Shape [equation_length, key_size].
    
    Returns:
        torch.Tensor: Float Tensor that indicates the scores under given information. Shape will be [batch_size, equation_length, ?]
    """
    output = encoded
    keys = sorted(modules.keys())

    # Apply modules (ascending order of keys).
    for key in keys:
        layer = modules[key]
        if isinstance(layer, (EPTMultiHeadAttention, EPTMultiHeadAttentionWeights)):
            output = layer(query=output, **kwargs)
        else:
            output = layer(output)

    return output

def apply_across_dim(function, dim=1, shared_keys=None, **tensors):# -> Dict[str, torch.Tensor]:
    """
    Apply a function repeatedly for each tensor slice through the given dimension.
    For example, we have tensor [batch_size, X, input_sequence_length] and dim = 1, then we will concatenate the following matrices on dim=1.
    - function([:, 0, :])
    - function([:, 1, :])
    - ...
    - function([:, X-1, :]).

    Args:
        function (function): Function to apply.
        dim (int): Dimension through which we'll apply function. (1 by default)
        shared_keys (set): Set of keys representing tensors to be shared. (None by default)
        tensors (torch.Tensor): Keyword arguments of tensors to compute. Dimension should >= `dim`.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary of tensors, whose keys are corresponding to the output of the function.
    """
    # Separate shared and non-shared tensors
    shared_arguments = {}
    repeat_targets = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or (shared_keys and key in shared_keys):
            shared_arguments[key] = tensor
        else:
            repeat_targets[key] = tensor

    # Check whether the size of the given dimension is the same across sliced_tensors.
    size = {key: tensor.shape[dim] for key, tensor in repeat_targets.items()}
    assert len(set(size.values())) == 1, 'Tensors does not have same size on dimension %s: We found %s' % (dim, size)

    # Since the sizes are the same, we will represent the size using the first entry.
    size = list(size.values())[0]

    # Dictionary for storing outputs
    output = {}

    for i in range(size):
        # Build kwargs for the function.
        kwargs = {key: tensor.select(dim=dim, index=i).contiguous() for key, tensor in repeat_targets.items()}
        kwargs.update(shared_arguments)

        # Apply function on the slice and restore the dimension for concatenation.
        for key, tensor in function(**kwargs).items():
            if key in shared_keys:
                continue

            if key not in output:
                output[key] = []

            output[key].append(tensor.unsqueeze(dim=dim))

    # Check whether the outputs are have the same size.
    assert all(len(t) == size for t in output.values())

    # Concatenate all outputs, and return.
    return {key: torch.cat(tensor, dim=dim).contiguous() for key, tensor in output.items()}



def get_embedding_without_pad(embedding: Union[nn.Embedding, torch.Tensor],
                              tokens: torch.Tensor, ignore_index=-1):# -> torch.Tensor:
    """
    Get embedding vectors of given token tensor with ignored indices are zero-filled.

    Args:
        embedding (nn.Embedding): An embedding instance
        tokens (torch.Tensor): A Long Tensor to build embedding vectors.
        ignore_index (int): Index to be ignored. `PAD_ID` by default.
    
    Returns:
        torch.Tensor: Embedding vector of given token tensor.
    """
    # Clone tokens and fill masked values as zeros.
    tokens = tokens.clone()
    ignore_positions = (tokens == ignore_index)
    if ignore_positions.any():
        tokens.masked_fill_(ignore_positions, 0)

    # Apply embedding matrix

    if isinstance(embedding, nn.Embedding):
        embedding = embedding(tokens)
    else:
        embedding = F.embedding(tokens, embedding)

    # Set masked values as zero vector.
    if ignore_positions.any():
        embedding.masked_fill_(ignore_positions.unsqueeze(-1), 0.0)

    return embedding.contiguous()
class LogSoftmax(nn.LogSoftmax):
    """
    LogSoftmax layer that can handle infinity values.
    """

    def forward(self, tensor: torch.Tensor):# -> torch.Tensor:
        """
        Compute log(softmax(tensor))

        Args:
            tensor torch.Tensor: FloatTensor whose log-softmax value will be computed
        
        Returns:
            torch.FloatTensor: LogSoftmax result.
        """
        # Find maximum values
        max_t = tensor.max(dim=self.dim, keepdim=True).values
        # Reset maximum as zero if it is a finite value.
        tensor = (tensor - max_t.masked_fill(~torch.isfinite(max_t), 0.0))

        # If a row's elements are all infinity, set the row as zeros to avoid NaN.
        all_inf_mask = torch.isinf(tensor).all(dim=self.dim, keepdim=True)
        if all_inf_mask.any().item():
            tensor = tensor.masked_fill(all_inf_mask, 0.0)

        # Forward nn.LogSoftmax.
        return super().forward(tensor)

def mask_forward(sz: int, diagonal: int = 1):# -> torch.Tensor:
    """
    Generate a mask that ignores future words. Each (i, j)-entry will be True if j >= i + diagonal

    Args:
        sz (int): Length of the sequence.
        diagonal (int): Amount of shift for diagonal entries.
    
    Returns: 
        torch.Tensor: Mask tensor with shape [sz, sz].
    """
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal=diagonal).contiguous()


class DecoderModel(nn.Module):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, config):
        """
        Initiate Equation Builder instance

        :param ModelConfig config: Configuration of this model
        """
        super().__init__()
        # Save configuration.
        self.config = config
        self.embedding_dim = 128 #self.config["embedding_dim"]
        self.hidden_dim = 768 #self.config["hidden_dim"]
        self.intermediate_dim = 3072
        self.num_decoder_layers = self.config["num_decoder_layers"]#6
        self.layernorm_eps = 1e-12
        self.num_decoder_heads = 12
        self.num_pointer_heads = self.config["num_pointer_heads"]#1
        self.num_hidden_layers = 6
        self.max_arity = 2

        self.training = True

    def init_factor(self):# -> float:
        """
        Returns:
            float: Standard deviation of normal distribution that will be used for initializing weights.
        """
        return 0.02

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        raise NotImplementedError()

    @property
    def is_expression_type(self) -> bool:
        """
        :rtype: bool
        :return: True if this model requires Expression type sequence
        """
        return self.required_field in ['ptr', 'gen']



    def _init_weights(self, module: nn.Module):
        """
        Initialize weights

        :param nn.Module module: Module to be initialized.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            # nn.Linear has 'weight' and 'bias', nn.Embedding has 'weight',
            # and nn.MultiheadAttention has *_weight and *_bias
            for name, param in module.named_parameters():
                if param is None:
                    continue

                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    param.data.zero_()
                else:
                    raise NotImplementedError("This case is not considered!")
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization as an identity funciton.
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_single(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of computed values
        """
        raise NotImplementedError()

    def _build_target_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
        """
        raise NotImplementedError()

    def forward(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        text_numpad:torch.Tensor = None, equation: torch.Tensor = None, beam: int = 1, max_len: int = 128,
                function_arities: Dict[int, int] = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of decoder model

        Returns:
            Dict[str, torch.Tensor]: Dictionary of tensors.
                If this model is currently on training phase, values will be accuracy or loss tensors
                Otherwise, values will be tensors representing predicted distribution of output
        """

        if equation is not None:
            output = self._forward_single(text, text_pad, text_num, text_numpad, equation)
            with torch.no_grad():
                targets = self._build_target_dict(equation, text_numpad)
            return output, targets
        else:
            self.training = False
            if 'expr' in self.config["decoder"]:
                batch_sz = text.shape[0]
                batch_range = range(batch_sz)
                device = text.device
                arity = self.max_arity

                if 'gen' in self.config["decoder"]:
                    num_range = lambda n: 1 <= n < 1 + EPT.NUM_MAX
                    con_range = lambda n: n == 0 or 1 + EPT.NUM_MAX + EPT.MEM_MAX <= n
                    # This type treats all operands as constants (no offsets)
                    num_offset = mem_offset = con_offset = 0
                else:
                    # This type dynamically concatenates all source of operands.
                    con_offset = 0
                    num_offset = self.constant_vocab_size
                    mem_offset = num_offset + text_num.shape[1]

                    con_range = lambda n: n < num_offset
                    num_range = lambda n: num_offset <= n < mem_offset


                function_arities = {} if self.function_arities is None else self.function_arities
                init = [EPT.FUN_NEW_EQN_ID] + [EPT.PAD_ID] * (2 * arity)

                result = torch.tensor([[[init]] for _ in batch_range], dtype=torch.long)

                # Prepare storage for beam scores. [B, M=1]
                beamscores = torch.zeros(batch_sz, 1)

                # Prepare indicator for termination
                all_exit = False
                seq_len = 1

                while seq_len < max_len and not all_exit:
                    # Compute scores for operator/operands
                    kwargs = {"text":text, "text_pad":text_pad, "text_num":text_num, "text_numpad":text_numpad, "equation":result.to(device)}
                    scores = apply_across_dim(self._forward_single, dim=1, shared_keys={"text", "text_pad", "text_num","text_numpad"},**kwargs)

                    #scores = self._forward_single(text, text_pad, text_num, text_numpad, equation=result.to(device))
                    # Retrieve score of the last token. [B, M, T, ?] -> [B, M, ?]
                    scores = {key: score[:, :, -1].cpu().detach() for key, score in scores.items()}

                    # Probability score for each beam & function words. [B, M, V] + [B, M, 1] = [B, M, V]
                    beam_function_score = scores['operator'] + beamscores.unsqueeze(-1)

                    # Prepare storage for the next results
                    next_beamscores = torch.zeros(batch_sz, beam)
                    next_result = torch.full((batch_sz, beam, seq_len + 1, 1 + 2 * arity), fill_value=-1,
                                             dtype=torch.long)

                    beam_range = range(beam_function_score.shape[1])
                    operator_range = range(beam_function_score.shape[2])
                    for i in batch_range:
                        # Compute scores for all (Beam, Operator, Operand) combinations. We will add all log probabilities.
                        # Storage for i-th item in the batch
                        score_i = []
                        for m in beam_range:
                            # For each beam, compute scores
                            # Check whether this beam was terminated before this step.
                            last_item = result[i, m, -1, 0].item()
                            after_last = last_item in {EPT.PAD_ID, EPT.FUN_END_EQN_ID}

                            if after_last:
                                # Score should be unchanged after __END_EQN token.
                                score_i.append((beamscores[i, m].item(), m, EPT.PAD_ID, []))
                                continue

                            # Compute beams for operators first.
                            operator_scores = {}
                            for f in operator_range:
                                operator_score = beam_function_score[i, m, f].item()

                                if f < len(EPT.FUN_TOKENS):
                                    if f == EPT.FUN_END_EQN_ID and last_item == EPT.FUN_NEW_EQN_ID:
                                        # Don't permit sequence like [__NEW_EQN, __END_EQN]
                                        continue

                                    # __NEW_EQN, __END_EQN, __NEW_VAR token does not require any arguments.
                                    score_i.append((operator_score, m, f, []))
                                else:
                                    operator_scores[f] = operator_score

                            # Combine operand log-probabilities with operator word log-probability.
                            operand_beams = [(0.0, [])]
                            for a in range(arity):
                                # Get top-k result
                                score_ia, index_ia = scores['operand_%s' % a][i, m].topk(beam)
                                score_ia = score_ia.tolist()
                                index_ia = index_ia.tolist()

                                # Compute M*M combination and preserve only top-M results.
                                operand_beams = [(s_prev + s_a, arg_prev + [arg_a])
                                                 for s_prev, arg_prev in operand_beams
                                                 for s_a, arg_a in zip(score_ia, index_ia)]
                                operand_beams = sorted(operand_beams, key=lambda t: t[0], reverse=True)[:beam]

                                for f, s_f in operator_scores.items():
                                    # Append expression (pair of operator and operands) that match current arity.
                                    if function_arities.get(f, arity) == a + 1:
                                        score_i += [(s_f + s_args, m, f, args) for s_args, args in operand_beams]

                        # Prepare the next beams. Scores[i] originally have shape [M, T] -> [M * T] after flattening.
                        beam_registered = set()
                        for score, prevbeam, operator, operands in sorted(score_i, key=lambda t: t[0], reverse=True):
                            if len(beam_registered) == beam:
                                # If beam was full, exit loop.
                                break

                            # Check whether this combination was already checked
                            beam_signature = (prevbeam, operator, *operands)
                            if beam_signature in beam_registered:
                                continue

                            # Set the next-beam
                            newbeam = len(beam_registered)
                            next_beamscores[i, newbeam] = score

                            # Copy tokens
                            next_result[i, newbeam, :-1] = result[i, prevbeam]
                            new_tokens = [operator]
                            for j, a in enumerate(operands):
                                # Assign operands and its source types.
                                if con_range(a):
                                    new_tokens += [EPT.ARG_CON_ID, a - con_offset]
                                elif num_range(a):
                                    new_tokens += [EPT.ARG_NUM_ID, a - num_offset]
                                else:
                                    new_tokens += [EPT.ARG_MEM_ID, a - mem_offset]
                            new_tokens = torch.as_tensor(new_tokens, dtype=torch.long, device=device)
                            next_result[i, newbeam, -1, :new_tokens.shape[0]] = new_tokens

                            # Assign beam information
                            beam_registered.add(beam_signature)

                    # Copy score information
                    beamscores = next_beamscores

                    # Update checks for termination
                    last_tokens = next_result[:, :, -1, 0]
                    all_exit = ((last_tokens == EPT.PAD_ID) | (last_tokens == EPT.FUN_END_EQN_ID)).all().item()

                    result = next_result
                    seq_len += 1
            else:
                batch_sz = text.shape[0]
                batch_range = range(batch_sz)
                device = text.device

                # Prepare inputs.
                # At the beginning, we start with only one beam, [B, M=1, T=1].
                result = torch.tensor([[[EPT.SEQ_NEW_EQN_ID]] for _ in batch_range], dtype=torch.long)

                # Prepare storage for beam scores. [B, M=1]
                beamscores = torch.zeros(batch_sz, 1)

                # Prepare indicator for termination
                all_exit = False
                seq_len = 1

                while seq_len < max_len and not all_exit:
                    # Compute scores
                    # Retrieve score of the last token. [B, M, T, ?] -> [B, M, ?]
                    scores = self._forward_single(text, text_pad, text_num, text_numpad, equation=result.to(device))
                    scores = scores['op'][:, :, -1].cpu().detach()

                    # Probability score for each beam & token. [B, M, V] + [B, M, 1] = [B, M, V]
                    beam_token_score = scores + beamscores.unsqueeze(-1)

                    # Prepare storage for the next results
                    next_beamscores = torch.zeros(batch_sz, beam)
                    next_result = torch.full((batch_sz, beam, seq_len + 1), fill_value=EPT.PAD_ID, dtype=torch.long)

                    beam_range = range(beam_token_score.shape[1])
                    token_range = range(beam_token_score.shape[2])
                    for i in batch_range:
                        # Compute scores for all (Beam, OpToken) combinations. We will add all log probabilities.
                        # Storage for i-th item in the batch
                        score_i = []
                        for m in beam_range:
                            # For each beam, compute scores
                            # Check whether this beam was terminated before this step.
                            last_item = result[i, m, -1].item()
                            after_last = last_item == EPT.PAD_ID or last_item == EPT.SEQ_END_EQN_ID

                            if after_last:
                                # Score should be unchanged after __END_EQN token.
                                score_i.append((beamscores[i, m].item(), m, EPT.PAD_ID))
                                continue

                            for v in token_range:
                                if v == EPT.SEQ_END_EQN_ID and last_item == EPT.SEQ_NEW_EQN_ID:
                                    # Don't permit sequence like [__NEW_EQN, __END_EQN]
                                    continue

                                token_score = beam_token_score[i, m, v].item()
                                score_i.append((token_score, m, v))

                        # Prepare the next beams. Scores[i] originally have shape [M, T] -> [M * T] after flattening.
                        beam_registered = set()
                        for score, prevbeam, token in sorted(score_i, key=lambda t: t[0], reverse=True):
                            if len(beam_registered) == beam:
                                # If beam was full, exit loop.
                                break

                            if (prevbeam, token, token) in beam_registered:
                                # If this combination was already checked, do not add this.
                                continue

                            # Set the next-beam
                            newbeam = len(beam_registered)
                            next_beamscores[i, newbeam] = score

                            # Copy tokens
                            next_result[i, newbeam, :-1] = result[i, prevbeam]
                            next_result[i, newbeam, -1] = token

                            # Assign beam information
                            beam_registered.add((prevbeam, token, token))

                    # Copy score information
                    beamscores = next_beamscores

                    # Update checks for termination
                    last_token_ids = next_result[:, :, -1]
                    all_exit = ((last_token_ids == EPT.PAD_ID) | (last_token_ids == EPT.SEQ_END_EQN_ID)).all().item()

                    result = next_result
                    seq_len += 1

            return result, None


class ExpressionDecoderModel(DecoderModel):
    """
    Decoding model that generates expression sequences (Abstract class)
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config)

        self.operator_vocab_size = len(out_idx2opsym)
        print(out_idx2opsym)
        self.operand_vocab_size = len(out_idx2consym)
        self.constant_vocab_size = len(out_idx2consym)
        self.max_arity = max([op['arity'] for op in EPT.OPERATORS.values()], default=2)

        self.function_arities = {i: EPT.OPERATORS[f]['arity'] for i, f in enumerate(out_idx2opsym) if i >= len(EPT.FUN_TOKENS)}
        """ Embedding layers """
        # Look-up table E_f(.) for operator embedding vectors (in Equation 2)
        self.operator_word_embedding = nn.Embedding(self.operator_vocab_size, self.hidden_dim)
        # Positional encoding PE(.) (in Equation 2, 5)
        self.operator_pos_embedding = EPTPositionalEncoding(self.hidden_dim)
        # Vectors representing source: u_num, u_const, u_expr in Equation 3, 4, 5
        self.operand_source_embedding = nn.Embedding(3, self.hidden_dim)

        """ Scalar parameters """
        # Initial degrading factor value for c_f and c_a.
        degrade_factor = self.embedding_dim ** 0.5
        # c_f in Equation 2
        self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        # c_a in Equation 3, 4, 5
        self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Layer Normalizations """
        # LN_f in Equation 2
        self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        # LN_a in Equation 3, 4, 5
        self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        """ Linear Transformation """
        # Linear transformation from embedding space to hidden space: FF_in in Equation 1.
        self.embed_to_hidden = nn.Linear(self.hidden_dim * (self.max_arity + 1), self.hidden_dim)

        """ Transformer layer """
        # Shared transformer layer for decoding (TransformerDecoder in Figure 2)
        self.shared_decoder_layer = EPTTransformerLayer(hidden_dim = self.hidden_dim, num_decoder_heads = self.num_decoder_heads, layernorm_eps = self.layernorm_eps,intermediate_dim= self.intermediate_dim)

        """ Output layer """
        # Linear transformation from hidden space to pseudo-probability space: FF_out in Equation 6
        self.operator_out = nn.Linear(self.hidden_dim, self.operator_vocab_size)
        # Softmax layers, which can handle infinity values properly (used in Equation 6, 10)
        self.softmax = LogSoftmax(dim=-1)

        # Argument output will be defined in sub-classes
        # Initialize will be done in sub-classes


    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor ids:
            LongTensor containing index-type information of operands. (This corresponds to a_ij in the paper)
        :param torch.Tensor mem_pos:
            FloatTensor containing positional encoding used so far. (i.e. PE(.) in the paper)
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
        :rtype: torch.Tensor
        :return: A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Compute input of the decoder

        Args:
            ids (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size]
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
        
        Returns:
            torch.Tensor: A FloatTensor representing input vector. Shape [batch_size, equation_length, hidden_size].
        """
        # Operator embedding: [B, T, H] (Equation 2)
        # - compute E_f first
        operator = get_embedding_without_pad(self.operator_word_embedding, ids.select(dim=-1, index=0))
        # - compute PE(.): [T, H]
        operator_pos = self.operator_pos_embedding(ids.shape[1])
        # - apply c_f and layer norm, and reshape it as [B, T, 1, H]
        operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0)).unsqueeze(2)

        # Operand embedding [B, T, A, H] (Equation 3, 4, 5)
        # - compute c_a u_* first.
        operand = get_embedding_without_pad(self.operand_source_embedding, ids[:, :, 1::2]) * self.operand_source_factor
        # - add operand embedding
        operand += self._build_operand_embed(ids, operator_pos, nums)
        # - apply layer norm
        operand = self.operand_norm(operand)

        # Concatenate embedding: [B, T, 1+A, H] -> [B, T, (1+A)H]

        operator_operands = torch.cat([operator, operand], dim=2).contiguous().flatten(start_dim=2)
        # Do linear transformation (Equation 1)
        return self.embed_to_hidden(operator_operands)

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor = None,
                               text: torch.Tensor = None, text_pad: torch.Tensor = None):# -> torch.Tensor:
        """
        Compute decoder's hidden state vectors

        Args:
            embedding (torch.Tensor): FloatTensor containing input vectors. Shape [batch_size, equation_length, hidden_size],
            embedding_pad (torch.Tensor):BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence, Shape [batch_size, equation_length]
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
        
        Returns: 
            torch.Tensor: A FloatTensor of shape [batch_size, equation_length, hidden_size], which contains decoder's hidden states.
        """
        # Build forward mask
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        # Repeatedly pass TransformerDecoder layer
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_decoder_layer(target=output, memory=text, target_attention_mask=mask,
                                               target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)

        return output

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        text_numpad:torch.Tensor = None, equation: torch.Tensor = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands.
                Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators. FloatTensor with shape [batch_size, equation_length, operator_size].
                '_out': Decoder's hidden states. FloatTensor with shape [batch_size, equation_length, hidden_size].
                '_not_usable': Indicating positions that corresponding output values are not usable in the operands. BoolTensor with Shape [batch_size, equation_length].
        """
        # Embedding: [B, T, H]
        operator_ids = equation.select(dim=2, index=0)
        output = self._build_decoder_input(ids=equation, nums=text_num)
        output_pad = operator_ids == EPT.PAD_ID

        # Ignore the result of equality at the function output
        output_not_usable = output_pad.clone()
        output_not_usable[:, :-1].masked_fill_(operator_ids[:, 1:] == EPT.FUN_EQ_SGN_ID, True)
        # We need offset '1' because 'function_word' is input and output_not_usable is 1-step shifted output.

        # Decoder output: [B, T, H]
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)

        # Compute function output (with 'NEW_EQN' masked)
        operator_out = self.operator_out(output)

        if not self.training:

            operator_out[:, :, EPT.FUN_NEW_EQN_ID] = EPT.NEG_INF
            # Can end after equation formed, i.e. END_EQN is available when the input is EQ_SGN.
            operator_out[:, :, EPT.FUN_END_EQN_ID].masked_fill_(operator_ids != EPT.FUN_EQ_SGN_ID, EPT.NEG_INF)


        # Predict function output.
        result = {'operator': self.softmax(operator_out),
                    '_out': output, '_not_usable': output_not_usable}

        # Remaining work will be done by subclasses
        return result


class ExpressionTransformer(ExpressionDecoderModel):
    """
    Vanilla Transformer + Expression (The second ablated model)
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym)

        """ Operand embedding """
        # Look-up table for embedding vectors: E_c used in Vanilla Transformer + Expression (Appendix)
        self.operand_word_embedding = nn.Embedding(self.operand_vocab_size, self.hidden_dim)

        """ Output layer """
        # Linear transformation from hidden space to operand output space:
        # FF_j used in Vanilla Transformer + Expression (Appendix)
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_out': nn.Linear(self.hidden_dim, self.operand_vocab_size)
            }) for _ in range(self.max_arity)
        ])

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return 'gen'

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Build operand embedding.

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length, 1+2*arity_size].
            mem_pos (torch.Tensor): FloatTensor containing positional encoding used so far. Shape [batch_size, equation_length, hidden_size], where hidden_size = dimension of hidden state
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing operand embedding vector. Shape [batch_size, equation_length, arity_size, hidden_size]
        """
        # Compute operand embedding (Equation 4 in the paper and 3-rd and 4-th Equation in the appendix)
        # Adding u vectors will be done in _build_decoder_input.
        # We will ignore information about the source (slice 1::2) and operator (index 0).
        return get_embedding_without_pad(self.operand_word_embedding, ids[:, :, 2::2])

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None,
                        text_num: torch.Tensor = None, text_numpad: torch.Tensor = None,
                        equation: torch.Tensor = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text.Shape: [batch_size, num_size, hidden_size].
            text_numpad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators. FloatTensor with shape [batch_size, equation_length, operator_size], where operator_size = size of operator vocabulary.
                'operand_J': Log probability of next J-th operands.FloatTensor with shape [batch_size, equation_length, operand_size].
        """
        # Retrieve decoder's hidden states
        # Dictionary will have 'func', '_out', and '_not_usable'
        result = super()._forward_single(text, text_pad, text_num, text_numpad, equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden state: [B, T, H]
        output = result.pop('_out')
        # Mask indicating whether expression can be used as an operand: [B, T] -> [B, 1, T]
        output_not_usable = result.pop('_not_usable').unsqueeze(1)
        # Forward mask: [T, T] -> [1, T, T]
        forward_mask = mask_forward(output.shape[1], diagonal=0).unsqueeze(0).to(output.device)

        # Number tokens are placed on 1:1+NUM_MAX
        num_begin = 1
        num_used = num_begin + min(text_num.shape[1], EPT.NUM_MAX)
        num_end = num_begin + EPT.NUM_MAX
        # Memory tokens are placed on 1+NUM_MAX:1+NUM_MAX+MEM_MAX
        mem_used = num_end + min(output.shape[1], EPT.MEM_MAX)
        mem_end = num_end + EPT.MEM_MAX

        # Predict arguments
        for j, layer in enumerate(self.operand_out):
            word_output = apply_module_dict(layer, encoded=output)

            # Mask probabilities when evaluating.
            if not self.training:
                # Ignore probabilities on not-appeared number tokens
                word_output[:, :, num_begin:num_used].masked_fill_(text_numpad.unsqueeze(1), EPT.NEG_INF)
                word_output[:, :, num_used:num_end] = EPT.NEG_INF

                # Ignore probabilities on non-appeared memory tokens
                word_output[:, :, num_end:mem_used].masked_fill_(output_not_usable, EPT.NEG_INF)
                word_output[:, :, num_end:mem_used].masked_fill_(forward_mask, EPT.NEG_INF)
                word_output[:, :, mem_used:mem_end] = EPT.NEG_INF

            # Apply softmax after masking (compute 'operand_J')
            result['operand_%s' % j] = self.softmax(word_output)

        return result
    def _build_target_dict(self, equation, num_pad=None):# -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of target values
                'operator': Index of next operators. LongTensor with shape [batch_size, equation_length].
                'operand_J': Index of next J-th operands. LongTensor with shape [batch_size, equation_length].
        """
        # Build targets

        targets = {'operator': equation.select(dim=-1, index=0)}
        for j in range(2):
            targets['operand_%s' % j] = equation[:, :, (j * 2 + 2)]

        return targets




class ExpressionPointerTransformer(ExpressionDecoderModel):
    """
    The EPT model
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym)

        """ Operand embedding """
        # Look-up table for constants: E_c used in Equation 4
        self.constant_word_embedding = nn.Embedding(self.constant_vocab_size, self.hidden_dim)

        """ Output layer """
        # Group of layers to compute Equation 8, 9, and 10
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_attn': EPTMultiHeadAttentionWeights(hidden_dim=self.hidden_dim, num_heads=self.num_pointer_heads),
                '1_mean': Squeeze(dim=-1) if self.num_pointer_heads == 1 else AveragePooling(dim=-1)
            }) for _ in range(self.max_arity)
        ])

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return "ptr"

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Build operand embedding.

        Args: 
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length, 1+2*arity_size].
            mem_pos (torch.Tensor): FloatTensor containing positional encoding used so far. Shape [batch_size, equation_length, hidden_size].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing operand embedding vector. Shape [batch_size, equation_length, arity_size, hidden_size]
        """
        # Tensor ids has 1 vocabulary index of operator and A pair of (source of operand, vocabulary index of operand)
        # Source of operand (slice 1::2), shape [B, T, A]
        operand_source = ids[:, :, 1::2]
        # Index of operand (slice 2::2), shape [B, T, A]
        operand_value = ids[:, :, 2::2]

        # Compute for number operands: [B, T, A, E] (Equation 3)
        number_operand = operand_value.masked_fill(operand_source != EPT.ARG_NUM_ID, EPT.PAD_ID)
        operand = torch.stack([get_embedding_without_pad(nums[b], number_operand[b])
                               for b in range(ids.shape[0])], dim=0).contiguous()

        # Compute for constant operands: [B, T, A, E] (Equation 4)
        operand += get_embedding_without_pad(self.constant_word_embedding,
                                             operand_value.masked_fill(operand_source != EPT.ARG_CON_ID, EPT.PAD_ID))

        # Compute for prior-result operands: [B, T, A, E] (Equation 5)
        prior_result_operand = operand_value.masked_fill(operand_source != EPT.ARG_MEM_ID, EPT.PAD_ID)
        operand += get_embedding_without_pad(mem_pos, prior_result_operand)
        return operand

    def _build_attention_keys(self, num: torch.Tensor, mem: torch.Tensor, num_pad: torch.Tensor = None,
                              mem_pad: torch.Tensor = None):# -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Attention Keys by concatenating all items.

        Args: 
            num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
            mem (torch.Tensor): FloatTensor containing decoder's hidden states corresponding to prior expression outputs. Shape [batch_size, equation_length, hidden_size].
            num_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            mem_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the target expression sequence. Shape [batch_size, equation_length]
        
        Returns: 
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Triple of Tensors
                - [0] Keys (A_ij in the paper). Shape [batch_size, constant_size+num_size+equation_length, hidden_size], where C = size of constant vocabulary.
                - [1] Mask for positions that should be ignored in keys. Shape [batch_size, C+num_size+equation_length]
                - [2] Forward Attention Mask to ignore future tokens in the expression sequence. Shape [equation_length, C+num_size+equation_length]
        """
        # Retrieve size information
        batch_sz = num.shape[0]
        const_sz = self.constant_vocab_size
        const_num_sz = const_sz + num.shape[1]

        # Order: Const, Number, Memory
        # Constant keys: [C, E] -> [1, C, H] -> [B, C, H]
        const_key = self.constant_word_embedding.weight.unsqueeze(0).expand(batch_sz, const_sz, self.hidden_dim)

        # Key: [B, C+N+T, H]
        key = torch.cat([const_key, num, mem], dim=1).contiguous()
        # Key ignorance mask: [B, C+N+T]
        key_ignorance_mask = torch.zeros(key.shape[:2], dtype=torch.bool, device=key.device)
        if num_pad is not None:
            key_ignorance_mask[:, const_sz:const_num_sz] = num_pad
        if mem_pad is not None:
            key_ignorance_mask[:, const_num_sz:] = mem_pad

        # Attention mask: [T, C+N+T], exclude self.
        attention_mask = torch.zeros(mem.shape[1], key.shape[1], dtype=torch.bool, device=key.device)
        attention_mask[:, const_num_sz:] = mask_forward(mem.shape[1], diagonal=0).to(key_ignorance_mask.device)

        return key, key_ignorance_mask, attention_mask

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None,
                        text_num: torch.Tensor = None, text_numpad: torch.Tensor = None,
                        equation: torch.Tensor = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        Args: 
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
            text_numpad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators.FloatTensor with shape [batch_size, equation_length, operator_size].
                'operand_J': Log probability of next J-th operands. FloatTensor with shape [batch_size, equation_length, operand_size].
        """
        # Retrieve decoder's hidden states
        # Dictionary will have 'func', '_out', and '_not_usable'
        result = super()._forward_single(text, text_pad, text_num, text_numpad,equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden states: [B, T, H]
        output = result.pop('_out')
        # Mask indicating whether expression can be used as an operand: [B, T] -> [B, 1, T]
        output_not_usable = result.pop('_not_usable')

        # Build attention keys by concatenating constants, written numbers and prior outputs (Equation 7)
        key, key_ign_msk, attn_msk = self._build_attention_keys(num=text_num, mem=output,
                                                                num_pad=text_numpad, mem_pad=output_not_usable)

        # Predict arguments (Equation 8, 9, 10)
        for j, layer in enumerate(self.operand_out):
            score = apply_module_dict(layer, encoded=output, key=key, key_ignorance_mask=key_ign_msk,
                                      attention_mask=attn_msk)
            result['operand_%s' % j] = self.softmax(score)

        return result

    def _build_target_dict(self, equation, num_pad=None):# -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        Returns: 
            Dict[str, torch.Tensor]: Dictionary of target values
                'operator': Index of next operators. LongTensor with shape [batch_size, equation_length].
                'operand_J': Index of next J-th operands. LongTensor with shape [batch_size, equation_length].
        """
        # Build targets


        # Offset of written numbers
        num_offset = self.constant_vocab_size
        # Offset of prior expressions
        mem_offset = num_offset + num_pad.shape[1]

        # Build dictionary for targets
        targets = {'operator': equation.select(dim=-1, index=0)}
        for i in range(self.max_arity):
            # Source of the operand
            operand_source = equation[:, :, (i * 2 + 1)]
            # Value of the operand
            operand_value = equation[:, :, (i * 2 + 2)].clamp_min(0)

            # Add index offsets.
            # - PAD_ID will be PAD_ID (-1),
            # - constants will use range from 0 to C (number of constants; exclusive)
            # - numbers will use range from C to C + N (N = max_num)
            # - prior expressions will use range C + N to C + N + T
            operand_value += operand_source.masked_fill(operand_source == EPT.ARG_NUM_ID, num_offset) \
                .masked_fill_(operand_source == EPT.ARG_MEM_ID, mem_offset)

            # Assign target value of J-th operand.
            targets['operand_%s' % i] = operand_value

        return targets


class OpDecoderModel(DecoderModel):
    """
    Decoding model that generates Op(Operator/Operand) sequences (Abstract class)
    """

    def __init__(self, config):
        super().__init__(config)

        """ Embedding look-up tables """
        # Token-level embedding
        self.word_embedding = nn.Embedding(config["op_vocab_size"], config["hidden_dim"])
        # Positional encoding
        self.pos_embedding = EPTPositionalEncoding(config["hidden_dim"])
        # LayerNorm for normalizing word embedding vector.
        self.word_hidden_norm = nn.LayerNorm(config["hidden_dim"], eps=self.layernorm_eps)
        # Factor that upweights word embedding vector. (c_in in the Appendix)
        degrade_factor = config["hidden_dim"] ** 0.5
        self.pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Decoding layer """
        # Shared transformer layer for decoding
        self.shared_layer = EPTTransformerLayer(hidden_dim = self.hidden_dim, num_decoder_heads = self.num_decoder_heads, layernorm_eps = self.layernorm_eps,intermediate_dim= self.intermediate_dim)

        # Output layer will be defined in sub-classes
        # Weight will be initialized by sub-classes


    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Build Op embedding

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing op embedding vector. Shape [batch_size, equation_length, hidden_size]
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Compute input of the decoder.

        Args:
            ids (torch.Tensor): LongTensor containing op tokens. Shape: [batch_size, equation_length]
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size],
        
        Returns: 
            torch.Tensor: A FloatTensor representing input vector. Shape [batch_size, equation_length, hidden_size].
        """
        # Positions: [T, E]
        pos = self.pos_embedding(ids.shape[1])
        # Word embeddings: [B, T, E]
        word = self._build_word_embed(ids, nums)
        # Return [B, T, E]
        return self.word_hidden_norm(word * self.pos_factor + pos.unsqueeze(0))

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor = None,
                               text: torch.Tensor = None, text_pad: torch.Tensor = None):# -> torch.Tensor:
        """
        Compute decoder's hidden state vectors.

        Args: 
            embedding (torch.Tensor): FloatTensor containing input vectors. Shape [batch_size, decoding_sequence, input_embedding_size].
            embedding_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence. Shape [batch_size, decoding_sequence]
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
        
        Returns: 
        torch.Tensor: A FloatTensor of shape [batch_size, decoding_sequence, hidden_size], which contains decoder's hidden states.
        """
        # Build forward mask
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        # Repeatedly pass TransformerDecoder layer
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_layer(target=output, memory=text, target_attention_mask=mask,
                                       target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)

        return output

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        text_numpad:torch.Tensor = None, equation: torch.Tensor = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states e_i. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, input_embedding_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                '_out': Decoder's hidden states. FloatTensor with shape [batch_size, equation_length, hidden_size].
        """
        # Embedding: [B, T, H]
        output = self._build_decoder_input(ids=equation, nums=text_num.relu())
        output_pad = equation == EPT.PAD_ID

        # Decoder's hidden states: [B, T, H]
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        result = {'_out': output}

        # Remaining work will be done by subclasses
        return result


class VanillaOpTransformer(OpDecoderModel):
    """
    The vanilla Transformer model
    """

    def __init__(self, config):
        super().__init__(config)

        """ Op token Generator """
        self.op_out = nn.Linear(config["hidden_dim"], config["op_vocab_size"])
        self.softmax = LogSoftmax(dim=-1)

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return 'vallina'

    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor):# -> torch.Tensor:
        """
        Build Op embedding

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor:A FloatTensor representing op embedding vector. Shape [batch_size, equation_length, hidden_size].
        """
        return get_embedding_without_pad(self.word_embedding, ids)

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        text_numpad: torch.Tensor = None, equation: torch.Tensor = None):# -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, input_embedding_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                'op': Log probability of next op tokens. FloatTensor with shape [batch_size, equation_length, operator_size].
        """
        # Retrieve decoder's hidden states
        # Dictionary will have '_out'
        result = super()._forward_single(text, text_pad, text_num, equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden states: [B, T, H]
        output = result.pop('_out')

        # Predict the next op token: Shape [B, T, V].
        op_out = self.op_out(output)
        result['op'] = self.softmax(op_out)

        return result
    def _build_target_dict(self, equation, num_pad=None):# -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        Returns: 
            Dict[str, torch.Tensor]: Dictionary of target values
                'op': Index of next op tokens. LongTensor with shape [batch_size, equation_length].
        """
        # Build targets
        return {'op': equation}
