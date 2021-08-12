import random
import torch

from mwptoolkit.utils.enum_type import FixType, NumMask,SpecialTokens, EPT
from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.utils.preprocess_tools import find_ept_numbers_in_text, postfix_parser, pad_token_ept_inp, ept_equ_preprocess


from transformers import AutoTokenizer,BertTokenizer

class DataLoaderEPT(AbstractDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)

        if self.dataset in ['math23k','hmwp']:
            pretrained_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        else:
            self.pretrained_tokenzier = AutoTokenizer.from_pretrained(config["pretrained_model_path"])
            
        self.pretrained_tokenzier.add_special_tokens({'additional_special_tokens': ['[N]']})
        
        
        self.out_unk_token = dataset.out_symbol2idx[EPT.ARG_UNK]
        self.model = config["model"].lower()
        self.decoder = config["decoder"].lower()
    
    def load_data(self, type):
        if type == "train":
            datas = self.dataset.trainset
            batch_size = self.train_batch_size
        elif type == "valid":
            datas = self.dataset.validset
            batch_size = self.test_batch_size
        elif type == "test":
            datas = self.dataset.testset
            batch_size = self.test_batch_size
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))

        num_total = len(datas)
        batch_num = int(num_total / batch_size) + 1
        for batch_i in range(batch_num):
            start_idx = batch_i * batch_size
            end_idx = (batch_i + 1) * batch_size
            if end_idx <= num_total:
                batch_data = datas[start_idx:end_idx]
            else:
                batch_data = datas[start_idx:num_total]
            if batch_data != []:
                batch_data = self.load_batch(batch_data)
                yield batch_data
    

    def load_batch(self, batch_data):
        '''
        {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos,
                            "visible matrix":d["visible matrix"],"position":d["position"],"id":d["id"]}
        '''
    
        equ_tokens_batch = []
        ques_batch = []

        infix_equ_batch = []

        num_list_batch = []
        num_pos_batch = []

        id_batch = []
        ans_batch = []

        ques_mask_batch = []
        equ_mask_batch = []
        num_mask_batch = []

        equ_len_batch = []
        ques_len_batch = []

        num_size_batch = []
        num_stack_batch = []

        group_nums_batch = []
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
    
