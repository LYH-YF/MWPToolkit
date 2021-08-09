from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset
import torch

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from mwptoolkit.utils.enum_type import FixType, NumMask,SpecialTokens, EPT
from mwptoolkit.utils.preprocess_tools import find_ept_numbers_in_text, postfix_parser, pad_token_ept_inp, ept_equ_preprocess

def get_num_mask(num_size_batch, generate_nums):
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    return num_mask

class PretrainDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # self.dataset=PretrainDataset(config)
        # dataset=PretrainDataset(config)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.in_pad_token = dataset.tokenizer.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)
        self.in_unk_token = dataset.tokenizer.convert_tokens_to_ids(SpecialTokens.UNK_TOKEN)

        if self.symbol_for_tree or self.equation_fix == FixType.MultiWayTree:
            self.out_pad_token = self.in_pad_token
            self.out_unk_token = dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
            self.temp_unk_token = dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
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
    
    def load_batch(self,batch_data):
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
            data['question']=self.dataset.tokenizer.tokenize(' '.join(data["question"]))
        batch_data=sorted(batch_data,key=lambda x:len(x['question']),reverse=True)
        for data in batch_data:
            ques_tensor = []
            equ_tensor = []
            temp_tensor = []
            sentence = data["question"]
            equation = data["equation"]
            template = data["template"]

            # question word to index
            print(len(data["question"]))
            #sentence=self.dataset.tokenizer.tokenize(' '.join(data["question"]))
            if self.add_sos:
                sentence=[SpecialTokens.SOS_TOKEN]+sentence
            if self.add_eos:
                sentence=sentence+[SpecialTokens.EOS_TOKEN]
            ques_tensor=self._word2idx(sentence)

            # equation symbol to index
            if self.share_vocab:
                equation=self.dataset.tokenizer.tokenize(' '.join(data["equation"]))
                template=self.dataset.tokenizer.tokenize(' '.join(data["template"]))
            if self.symbol_for_tree or self.equation_fix == FixType.MultiWayTree:
                pass
            else:
                equation.append(SpecialTokens.EOS_TOKEN)
                template.append(SpecialTokens.EOS_TOKEN)
            equ_tensor = self._equ_symbol2idx(equation)
            temp_tensor = self._temp_symbol2idx(template)

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
        if self.max_len != None:
            ques_len_batch=[self.max_len if l>self.max_len else l for l in ques_len_batch]
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
            "infix equation": infix_equ_batch,
        }
    
    def _word2idx(self, sentence):
        sentence_idx = []
        
        sentence_idx = self.dataset.tokenizer.convert_tokens_to_ids(sentence)
        
        return sentence_idx
    
    def _equ_symbol2idx(self, equation):
        equ_idx = []
        if self.equation_fix == FixType.MultiWayTree:
            for symbol in equation:
                if isinstance(symbol, list):
                    sub_equ_idx = self._equ_symbol2idx(symbol)
                    equ_idx.append(sub_equ_idx)
                else:
                    if self.share_vocab:
                        idx = self.dataset.tokenizer.convert_tokens_to_ids(symbol)
                    else:
                        try:
                            idx = self.dataset.out_symbol2idx[symbol]
                        except:
                            idx = self.out_unk_token
                    equ_idx.append(idx)
        else:
            for symbol in equation:
                if self.share_vocab:
                    idx = self.dataset.tokenizer.convert_tokens_to_ids(symbol)
                    
                else:
                    try:
                        idx = self.dataset.out_symbol2idx[symbol]
                    except:
                        idx = self.out_unk_token
                equ_idx.append(idx)
        return equ_idx
