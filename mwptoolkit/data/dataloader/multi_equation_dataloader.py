import random
import torch

from mwptoolkit.utils.enum_type import FixType, NumMask,SpecialTokens, EPT
from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.utils.preprocess_tools import find_ept_numbers_in_text, postfix_parser, pad_token_ept_inp, ept_equ_preprocess


from transformers import AutoTokenizer

def get_num_mask(num_size_batch, generate_nums):
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    return num_mask


class MultiEquationDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)

        if config["pretrained_model"]:
            self.pretrained_tokenzier = AutoTokenizer.from_pretrained(config["pretrained_model"])
            if config["model"].lower() in ["ept"]:
                self.pretrained_tokenzier.add_special_tokens({'additional_special_tokens': ['[N]']})
        else:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
            self.in_unk_token = dataset.in_word2idx[SpecialTokens.UNK_TOKEN]

        if self.symbol_for_tree or self.equation_fix == FixType.MultiWayTree:
            self.out_pad_token = self.in_pad_token
            self.out_unk_token = dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
            self.temp_unk_token = dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
        elif config["model"].lower() in ["ept"]:
            self.out_unk_token = dataset.out_symbol2idx[EPT.ARG_UNK]
            self.model = config["model"].lower()
            self.decoder = config["decoder"].lower()
        else:
            if self.share_vocab:
                self.out_pad_token = self.in_pad_token
                self.out_unk_token = self.in_unk_token
                self.temp_pad_token = self.in_pad_token
                self.temp_unk_token = self.in_unk_token
            else:
                self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.out_unk_token = dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
                self.temp_pad_token = dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.temp_unk_token = dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]


    def _get_number_position(self, seq, num_list):
        num_pos = []
        for num in num_list:
            num_idx = self.dataset.in_word2idx[num]
            num_pos.append(seq.index(num_idx))
        return num_pos

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

    def load_batch_spans(self,batch_data):
        pad_num_pos = [-1] * len(self.dataset.out_idx2symbol)
        max_span_nums=0
        span_nums_batch=[]
        spans_batch=[]
        spans_length_batch=[]
        trees_batch=[]
        span_num_pos_batch=[]
        word_num_poses_batch=[]
        word_num_poses_pad_batch=[]
        for data in batch_data:
            span_nums = len(data['split sentences'])
            span_nums_batch.append(span_nums)
            span_num_pos = [-1] * len(self.dataset.out_idx2symbol)
            word_num_poses = [[-1] * len(self.dataset.out_idx2symbol) for _ in range(span_nums)]
            for i, span in enumerate(data['split sentences']):
                for j, word in enumerate(span):
                    if word in NumMask.number and word in self.dataset.out_idx2symbol:
                        class_index = self.dataset.out_symbol2idx[word]
                        span_num_pos[class_index] = i
                        word_num_poses[i][class_index] = j
            span_num_pos_batch.append(span_num_pos)
            word_num_poses_batch.append(word_num_poses)
        max_span_nums=max(span_nums_batch)
        for span_idx in range(max_span_nums):
            span_i_batch=[]
            span_i_length_batch=[]
            tree_i_batch=[]
            word_num_poses_pad = []
            for b_i,data in enumerate(batch_data):
                if span_idx>=span_nums_batch[b_i]:
                    span_i_batch.append([])
                    span_i_length_batch.append(0)
                    word_num_poses_pad.append(pad_num_pos)
                else:
                    sentence=data['split sentences'][span_idx]
                    sentence_idx=self._word2idx(sentence)
                    span_i_length=len(sentence_idx)
                    span_i_batch.append(sentence_idx)
                    span_i_length_batch.append(span_i_length)
                    word_num_poses_pad.append(word_num_poses_batch[b_i][span_idx])
                try:
                    tree=data['deprel tree'][span_idx]
                    tree_i_batch.append(tree)
                except:
                    tree_i_batch.append(None)
            span_i_batch=self._pad_input_batch(span_i_batch,span_i_length_batch)
            spans_batch.append(span_i_batch)
            spans_length_batch.append(span_i_length_batch)
            trees_batch.append(tree_i_batch)
            word_num_poses_pad_batch.append(word_num_poses_pad)
        
        return spans_batch,spans_length_batch,span_nums_batch,trees_batch,span_num_pos_batch,word_num_poses_pad_batch
    

    def load_batch(self, batch_data):
        '''
        {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos,
                            "visible matrix":d["visible matrix"],"position":d["position"],"id":d["id"]}
        '''
        if self.model.lower() in ['ept']:
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
                tokenized = self.pretrained_tokenzier.tokenize(text)
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
        ques_batch = []
        equ_batch = []
        temp_batch = []
        ques_source_batch = []
        equ_source_batch = []
        temp_source_batch = []
        ques_source_1_batch = []
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
            ques_tensor = []
            equ_tensor = []
            temp_tensor = []
            num_pos = []
            sentence = data["question"]
            equation = data["equation"]
            template = data["template"]

            # question word to index
            if self.add_sos:
                ques_tensor.append(self.dataset.in_word2idx["<SOS>"])
            ques_tensor += self._word2idx(sentence)
            if self.add_eos:
                ques_tensor.append(self.dataset.in_word2idx["<EOS>"])

            # equation symbol to index
            equ_tensor = self._equ_symbol2idx(equation)
            temp_tensor = self._temp_symbol2idx(template)
            if self.symbol_for_tree or self.equation_fix == FixType.MultiWayTree:
                pass
            else:
                if self.share_vocab:
                    equ_tensor.append(self.dataset.in_word2idx["<EOS>"])
                    temp_tensor.append(self.dataset.in_word2idx["<EOS>"])
                else:
                    equ_tensor.append(self.dataset.out_symbol2idx["<EOS>"])
                    temp_tensor.append(self.dataset.temp_symbol2idx["<EOS>"])

            equ_len_batch.append(len(equ_tensor))
            ques_len_batch.append(len(ques_tensor))
            ques_batch.append(ques_tensor)
            equ_batch.append(equ_tensor)
            temp_batch.append(temp_tensor)

            # question / equation to string
            ques_source = ' '.join(sentence)
            if self.equation_fix == FixType.MultiWayTree:
                equ_source = ' '
                temp_source = ' '
            else:
                equ_source = ' '.join(equation)
                temp_source = ' '.join(template)
            ques_source_batch.append(ques_source)
            equ_source_batch.append(equ_source)
            temp_source_batch.append(temp_source)
            ques_source_1_batch.append(data["ques source 1"])

            # infix equation
            infix_equ_batch.append(data["infix equation"])
            # quantity list
            num_list_batch.append(data["number list"])
            # quantity position
            if self.add_sos:
                num_pos = [pos + 1 for pos in data["number position"]]  # pos plus one because of adding <SOS> at the head of sentence
            else:
                num_pos = [pos for pos in data["number position"]]
            num_pos_batch.append(num_pos)
            # question id and answer
            id_batch.append(data["id"])
            ans_batch.append(data["ans"])

            try:
                group_nums_batch.append(data["group nums"])
            except:
                group_nums_batch.append([])
            num_stack_batch.append(self._build_num_stack(equation, data["number list"]))
        # padding batch question
        ques_batch = self._pad_input_batch(ques_batch, ques_len_batch)
        # padding batch equation
        if self.equation_fix == FixType.MultiWayTree:
            pass
        else:
            equ_batch = self._pad_output_batch(equ_batch, equ_len_batch)
            temp_batch = self._pad_output_batch(temp_batch, equ_len_batch)

        # question mask
        ques_mask_batch = self._get_mask(ques_len_batch)
        # equation mask
        equ_mask_batch = self._get_mask(equ_len_batch)
        # quantity count
        num_size_batch = [len(num_pos) for num_pos in num_pos_batch]
        # quantity mask
        num_mask_batch = get_num_mask(num_size_batch, self.dataset.generate_list)

        new_group_nums_batch = []
        for group_nums in group_nums_batch:
            new_group_nums = []
            for group_num in group_nums:
                new_group_num = []
                for pos in group_num:
                    if self.add_sos:
                        new_group_num.append(pos + 1)
                    else:
                        new_group_num.append(pos)
                new_group_nums.append(new_group_num)
            new_group_nums_batch.append(new_group_nums)

        # to tensor
        ques_tensor_batch = torch.tensor(ques_batch).to(self.device)
        if self.equation_fix == FixType.MultiWayTree:
            equ_tensor_batch = equ_batch
            temp_tensor_batch = temp_batch
        else:
            equ_tensor_batch = torch.tensor(equ_batch).to(self.device)
            temp_tensor_batch = torch.tensor(temp_batch).to(self.device)
        ques_mask_batch = torch.tensor(ques_mask_batch).to(self.device).bool()
        num_mask_batch = torch.tensor(num_mask_batch).to(self.device).bool()
        ques_len_batch = torch.tensor(ques_len_batch).long()
        equ_mask_batch = torch.tensor(equ_mask_batch).to(self.device).bool()
        
        if self.dataset.model.lower() in ['hms']:
            spans_batch,spans_length_batch,span_nums_batch,trees_batch,\
                span_num_pos_batch,word_num_poses_batch=self.load_batch_spans(batch_data)
            # to tensor
            spans_batch = [torch.tensor(span_i_batch).to(self.device) for span_i_batch in spans_batch]
            spans_length_batch = torch.tensor(spans_length_batch).long()
            span_nums_batch = torch.tensor(span_nums_batch).to(self.device)
            span_num_pos_batch = torch.LongTensor(span_num_pos_batch).to(self.device)
            word_num_poses_batch = [torch.LongTensor(word_num_pos).to(self.device) for word_num_pos in word_num_poses_batch]
            return {
                "spans":spans_batch,
                "spans len":spans_length_batch,
                "span nums":span_nums_batch,
                "deprel tree":trees_batch,
                "span num pos":span_num_pos_batch,
                "word num poses":word_num_poses_batch,
                "question": ques_tensor_batch,
                "equation": equ_tensor_batch,
                "template": temp_tensor_batch,
                "ques len": ques_len_batch,
                "equ len": equ_len_batch,
                "num list": num_list_batch,
                "num pos": num_pos_batch,
                "id": id_batch,
                "num mask": num_mask_batch,
                "ques mask": ques_mask_batch,
                "equ mask": equ_mask_batch,
                "num stack": num_stack_batch,
                "ans": ans_batch,
                "num size": num_size_batch,
                "ques_source": ques_source_batch,
                "equ_source": equ_source_batch,
                "temp_source": temp_source_batch,
                "ques source 1": ques_source_1_batch,
                "group nums": new_group_nums_batch,
                "infix equation": infix_equ_batch,
            }
        return {
            "question": ques_tensor_batch,
            "equation": equ_tensor_batch,
            "template": temp_tensor_batch,
            "ques len": ques_len_batch,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "id": id_batch,
            "num mask": num_mask_batch,
            "ques mask": ques_mask_batch,
            "equ mask": equ_mask_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "num size": num_size_batch,
            "ques_source": ques_source_batch,
            "equ_source": equ_source_batch,
            "temp_source": temp_source_batch,
            "ques source 1": ques_source_1_batch,
            "group nums": new_group_nums_batch,
            "infix equation": infix_equ_batch
        }