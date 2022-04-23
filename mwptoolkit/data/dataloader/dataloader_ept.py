# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:34:16
# @File: dataloader_ept.py


import math
import torch
from typing import List

from mwptoolkit.config import Config
from mwptoolkit.data.dataset.dataset_ept import DatasetEPT
from mwptoolkit.utils.enum_type import EPT
from mwptoolkit.data.dataloader.template_dataloader import TemplateDataLoader
from mwptoolkit.utils.preprocess_tools import find_ept_numbers_in_text, pad_token_ept_inp, ept_equ_preprocess


from transformers import AutoTokenizer,BertTokenizer


class DataLoaderEPT(TemplateDataLoader):
    """dataloader class for deep-learning model EPT

    """
    def __init__(self, config:Config, dataset:DatasetEPT):
        """

        :param config:
        :param dataset:

        expected that config includes these parameters below:

        dataset (str): dataset name.

        pretrained_model_path (str): road path of pretrained model.

        decoder (str): decoder module name.

        model (str): model name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.

        train_batch_size (int): the training batch size.

        test_batch_size (int): the testing batch size.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        max_len (int|None): max input length.

        add_sos (bool): add sos token at the head of input sequence.

        add_eos (bool): add eos token at the tail of input sequence.
        """
        super().__init__(config, dataset)

        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)

        if config["dataset"] in ['math23k','hmwp']:
            self.pretrained_tokenzier = BertTokenizer.from_pretrained(config["pretrained_model_path"])
        else:
            self.pretrained_tokenzier = AutoTokenizer.from_pretrained(config["pretrained_model_path"])
            
        self.pretrained_tokenzier.add_special_tokens({'additional_special_tokens': ['[N]']})
        
        
        self.out_unk_token = dataset.out_symbol2idx[EPT.ARG_UNK]
        self.model = config["model"].lower()
        self.decoder = config["decoder"].lower()
        self.__init_batches()

    def __build_batch(self, batch_data):
        """load one batch

        Args:
            batch_data (list[dict])
        
        Returns:
            loaded batch data (dict)
        """
    
        equ_tokens_batch = []
        ques_batch = []

        infix_equ_batch = []

        num_list_batch = []

        id_batch = []
        ans_batch = []

        equ_len_batch = []
        ques_len_batch = []

        for data in batch_data:
            text, numbers = find_ept_numbers_in_text(data['ept']['text'],True)
            equation = data['ept']['expr']
            equ_tokens = ept_equ_preprocess(equation, self.decoder)

            #preprocessed_text, num_pos, numbers = ept_preprocess_input(text, numbers)
            tokenized = self.pretrained_tokenzier.tokenize(text.strip())
            ques_tensor = self.pretrained_tokenzier.convert_tokens_to_ids(tokenized)
            ques_batch.append(ques_tensor)
            ques_len_batch.append(len(ques_tensor))
            equ_tokens_batch.append(equ_tokens)
            equ_len_batch.append(len(equ_tokens))
            num_list_batch.append(numbers)
            ans_batch.append(data['ept']['answer'])
            id_batch.append(data["id"])
        ques_source_batch = ques_batch
    
        equ_source_batch = equ_tokens_batch
        ques_batch, num_pos_batch = pad_token_ept_inp(ques_batch, self.pretrained_tokenzier, num_list_batch)
        ques_tensor_batch = torch.as_tensor([self.pretrained_tokenzier.convert_tokens_to_ids(tok) for tok in ques_batch]).to(self.device)
        pad_masks = ques_tensor_batch == self.pretrained_tokenzier.pad_token_id
        
        num_size_batch = [len(num_) for num_ in num_list_batch]

        num_pos_batch = torch.as_tensor(num_pos_batch).long().to(self.device)
        
        if 'vall' in self.decoder:
            max_len = max(len(item) for item in equ_tokens_batch) + 2
            padded_batch = []

            for item in equ_tokens_batch:
                # Convert item into IDs
                item = [self.dataset.out_symbol2idx.get(tok, EPT.SEQ_UNK_TOK_ID) if tok != EPT.PAD_ID else tok
                        for tok in item]

                # Build padded item
                padded_item = [EPT.SEQ_NEW_EQN_ID] + item + [EPT.SEQ_END_EQN_ID]
                padded_item += [EPT.PAD_ID] * max(0, max_len - len(padded_item))

                padded_batch.append(padded_item)
                equ_len_batch.append(len(padded_item))
            equ_tensor_batch = torch.as_tensor(padded_batch).to(self.device)
        else:
            max_len = max(len(item) for item in equ_tokens_batch) + 2  # 2 = BOE/EOE
            padded_batch = []
            padded_id_batch = []
            # Padding for no-operand functions (i.e. special commands)
            max_arity_pad = [(None, None)] * 2

            for item in equ_tokens_batch:
                padded_item = [(EPT.FUN_NEW_EQN, max_arity_pad)]

                for operator, operands in item:
                    # We also had to pad operands.
                    remain_arity = max(0, 2 - len(operands))

                    operands = operands + max_arity_pad[:remain_arity]

                    padded_item.append((operator, operands))

                padded_item.append((EPT.FUN_END_EQN, max_arity_pad))
                padded_item += [(None, max_arity_pad)] * max(0, max_len - len(padded_item))

                padded_batch.append(padded_item)
                expr_sentence = []
                for expression in padded_item:

                    operator, operand = expression
                    operator = EPT.PAD_ID if operator is None else self.dataset.out_opsym2idx[operator]
                    # Convert operands
                    new_operands = []
                    for src, a in operand:
                        # For each operand, we attach [Src, Value] after the end of new_args.
                        if src is None:
                            new_operands += [EPT.PAD_ID, EPT.PAD_ID]
                        else:
                            # Get the source
                            new_operands.append(EPT.ARG_TOKENS.index(src))
                            # Get the index of value
                            if src == EPT.ARG_CON or 'gen' in self.decoder:
                                # If we need to look up the vocabulary, then find the index in it.
                                new_operands.append(self.dataset.out_consym2idx.get(a, EPT.ARG_UNK_ID))
                            else:
                                # Otherwise, use the index information that is already specified in the operand.
                                new_operands.append(a)
                    expr_sentence.append([operator] + new_operands)

                padded_id_batch.append(expr_sentence)
                equ_len_batch.append(len(expr_sentence))
            equ_tensor_batch = torch.as_tensor(padded_id_batch).to(self.device)
        
        #ques_mask_batch = self._get_mask(ques_len_batch)
        # equation mask
        #equ_mask_batch = self._get_mask(equ_len_batch)
        # quantity count

        # quantity mask


        return {
            "question": ques_tensor_batch,
            "equation": equ_tensor_batch,
            "ques mask": pad_masks,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "max numbers": max(len(numbers) for numbers in num_list_batch),
            "num pos": num_pos_batch,
            "id": id_batch,
            "ans": ans_batch,
            "num size": num_size_batch,
            "ques_source": ques_source_batch,
            "equ_source": equ_source_batch,
            "infix equation": infix_equ_batch,
        }

    def __init_batches(self):
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        for set_type in ['train', 'valid', 'test']:
            if set_type == 'train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
            elif set_type == 'valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
            elif set_type == 'test':
                datas = self.dataset.testset
                batch_size = self.test_batch_size
            else:
                raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
            num_total = len(datas)
            batch_num = math.ceil(num_total / batch_size)
            for batch_i in range(batch_num):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                if end_idx <= num_total:
                    batch_data = datas[start_idx:end_idx]
                else:
                    batch_data = datas[start_idx:num_total]
                built_batch = self.__build_batch(batch_data)
                if set_type == 'train':
                    self.trainset_batches.append(built_batch)
                elif set_type == 'valid':
                    self.validset_batches.append(built_batch)
                elif set_type == 'test':
                    self.testset_batches.append(built_batch)
                else:
                    raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)

    def build_batch_for_predict(self, batch_data: List[dict]):
        raise NotImplementedError

    # def load_batch(self, batch_data):
    #     """load one batch
    #
    #     Args:
    #         batch_data (list[dict])
    #
    #     Returns:
    #         loaded batch data (dict)
    #     """
    #
    #     equ_tokens_batch = []
    #     ques_batch = []
    #
    #     infix_equ_batch = []
    #
    #     num_list_batch = []
    #     num_pos_batch = []
    #
    #     id_batch = []
    #     ans_batch = []
    #
    #     ques_mask_batch = []
    #     equ_mask_batch = []
    #     num_mask_batch = []
    #
    #     equ_len_batch = []
    #     ques_len_batch = []
    #
    #     num_size_batch = []
    #     num_stack_batch = []
    #
    #     group_nums_batch = []
    #     for data in batch_data:
    #         text, numbers = find_ept_numbers_in_text(data['ept']['text'], True)
    #         equation = data['ept']['expr']
    #         equ_tokens = ept_equ_preprocess(equation, self.decoder)
    #
    #         # preprocessed_text, num_pos, numbers = ept_preprocess_input(text, numbers)
    #         tokenized = self.pretrained_tokenzier.tokenize(text.strip())
    #         ques_tensor = self.pretrained_tokenzier.convert_tokens_to_ids(tokenized)
    #         ques_batch.append(ques_tensor)
    #         ques_len_batch.append(len(ques_tensor))
    #         equ_tokens_batch.append(equ_tokens)
    #         equ_len_batch.append(len(equ_tokens))
    #         num_list_batch.append(numbers)
    #         ans_batch.append(data['ept']['answer'])
    #         id_batch.append(data["id"])
    #     ques_source_batch = ques_batch
    #
    #     equ_source_batch = equ_tokens_batch
    #     ques_batch, num_pos_batch = pad_token_ept_inp(ques_batch, self.pretrained_tokenzier, num_list_batch)
    #     ques_tensor_batch = torch.as_tensor(
    #         [self.pretrained_tokenzier.convert_tokens_to_ids(tok) for tok in ques_batch]).to(self.device)
    #     pad_masks = ques_tensor_batch == self.pretrained_tokenzier.pad_token_id
    #
    #     num_size_batch = [len(num_) for num_ in num_list_batch]
    #
    #     num_pos_batch = torch.as_tensor(num_pos_batch).long().to(self.device)
    #
    #     if 'vall' in self.decoder:
    #         max_len = max(len(item) for item in equ_tokens_batch) + 2
    #         padded_batch = []
    #
    #         for item in equ_tokens_batch:
    #             # Convert item into IDs
    #             item = [self.dataset.out_symbol2idx.get(tok, EPT.SEQ_UNK_TOK_ID) if tok != EPT.PAD_ID else tok
    #                     for tok in item]
    #
    #             # Build padded item
    #             padded_item = [EPT.SEQ_NEW_EQN_ID] + item + [EPT.SEQ_END_EQN_ID]
    #             padded_item += [EPT.PAD_ID] * max(0, max_len - len(padded_item))
    #
    #             padded_batch.append(padded_item)
    #             equ_len_batch.append(len(padded_item))
    #         equ_tensor_batch = torch.as_tensor(padded_batch).to(self.device)
    #     else:
    #         max_len = max(len(item) for item in equ_tokens_batch) + 2  # 2 = BOE/EOE
    #         padded_batch = []
    #         padded_id_batch = []
    #         # Padding for no-operand functions (i.e. special commands)
    #         max_arity_pad = [(None, None)] * 2
    #
    #         for item in equ_tokens_batch:
    #             padded_item = [(EPT.FUN_NEW_EQN, max_arity_pad)]
    #
    #             for operator, operands in item:
    #                 # We also had to pad operands.
    #                 remain_arity = max(0, 2 - len(operands))
    #
    #                 operands = operands + max_arity_pad[:remain_arity]
    #
    #                 padded_item.append((operator, operands))
    #
    #             padded_item.append((EPT.FUN_END_EQN, max_arity_pad))
    #             padded_item += [(None, max_arity_pad)] * max(0, max_len - len(padded_item))
    #
    #             padded_batch.append(padded_item)
    #             expr_sentence = []
    #             for expression in padded_item:
    #
    #                 operator, operand = expression
    #                 operator = EPT.PAD_ID if operator is None else self.dataset.out_opsym2idx[operator]
    #                 # Convert operands
    #                 new_operands = []
    #                 for src, a in operand:
    #                     # For each operand, we attach [Src, Value] after the end of new_args.
    #                     if src is None:
    #                         new_operands += [EPT.PAD_ID, EPT.PAD_ID]
    #                     else:
    #                         # Get the source
    #                         new_operands.append(EPT.ARG_TOKENS.index(src))
    #                         # Get the index of value
    #                         if src == EPT.ARG_CON or 'gen' in self.decoder:
    #                             # If we need to look up the vocabulary, then find the index in it.
    #                             new_operands.append(self.dataset.out_consym2idx.get(a, EPT.ARG_UNK_ID))
    #                         else:
    #                             # Otherwise, use the index information that is already specified in the operand.
    #                             new_operands.append(a)
    #                 expr_sentence.append([operator] + new_operands)
    #
    #             padded_id_batch.append(expr_sentence)
    #             equ_len_batch.append(len(expr_sentence))
    #         equ_tensor_batch = torch.as_tensor(padded_id_batch).to(self.device)
    #
    #     # ques_mask_batch = self._get_mask(ques_len_batch)
    #     # equation mask
    #     # equ_mask_batch = self._get_mask(equ_len_batch)
    #     # quantity count
    #
    #     # quantity mask
    #
    #     return {
    #         "question": ques_tensor_batch,
    #         "equation": equ_tensor_batch,
    #         "ques mask": pad_masks,
    #         "equ len": equ_len_batch,
    #         "num list": num_list_batch,
    #         "max numbers": max(len(numbers) for numbers in num_list_batch),
    #         "num pos": num_pos_batch,
    #         "id": id_batch,
    #         "ans": ans_batch,
    #         "num size": num_size_batch,
    #         "ques_source": ques_source_batch,
    #         "equ_source": equ_source_batch,
    #         "infix equation": infix_equ_batch,
    #     }

