import random
import torch

from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
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
        self.trainset_nums=len(dataset.trainset)
        self.validset_nums=len(dataset.validset)
        self.testset_nums=len(dataset.testset)
    def _get_number_position(self,seq,num_list):
        num_pos=[]
        for num in num_list:
            num_idx=self.dataset.in_word2idx[num]
            num_pos.append(seq.index(num_idx))
        return num_pos
    
    def load_data(self,type):
        if type == "train":
            datas = self.dataset.trainset
            batch_size=self.train_batch_size
        elif type =="valid":
            datas = self.dataset.validset
            batch_size=self.test_batch_size
        elif type=="test":
            datas=self.dataset.testset
            batch_size=self.test_batch_size
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))

        num_total=len(datas)
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
        ques_batch = []
        equ_batch = []
        ques_source_batch=[]
        equ_source_batch=[]

        num_list_batch = []
        num_pos_batch = []
        
        id_batch = []
        ans_batch = []
        
        ques_mask_batch = []
        equ_mask_batch=[]
        num_mask_batch = []
        
        equ_len_batch = []
        ques_len_batch = []
        
        num_size_batch = []
        num_stack_batch = []
        for data in batch_data:
            ques_tensor = []
            equ_tensor = []
            num_pos=[]
            sentence = data["question"]
            equation = data["equation"]
            ques_source=''.join(sentence)
            #equ_source=''.join(equation)
            ques_source_batch.append(ques_source)
            equ_source_batch.append(equation)
            num_list_batch.append(data["number list"])
            #num_pos_batch.append(data["number position"])
            id_batch.append(data["id"])
            #ques_len_batch.append(len(data["question"]))
            ans_batch.append(data["ans"])
            num_stack_batch.append(
                self._build_num_stack(equation, data["number list"]))
            if self.symbol_for_tree:
                pass
            else:
                ques_tensor.append(self.dataset.in_word2idx["<SOS>"])
            for word in sentence:
                try:
                    idx = self.dataset.in_word2idx[word]
                except:
                    idx = self.in_unk_token
                ques_tensor.append(idx)
            ques_tensor.append(self.dataset.in_word2idx["<EOS>"])
            
            num_pos=[pos+1 for pos in data["number position"]]
            num_pos_batch.append(num_pos)

            for word in equation:
                if self.share_vocab:
                    try:
                        idx = self.dataset.in_word2idx[word]
                    except:
                        idx = self.in_unk_token
                else:
                    try:
                        idx = self.dataset.out_symbol2idx[word]
                    except:
                        idx = self.out_unk_token
                equ_tensor.append(idx)
            if self.symbol_for_tree:
                pass
            else:
                if self.share_vocab:
                    equ_tensor.append(self.dataset.in_word2idx["<EOS>"])
                else:
                    equ_tensor.append(self.dataset.out_symbol2idx["<EOS>"])
            
            equ_len_batch.append(len(equ_tensor))
            ques_len_batch.append(len(ques_tensor))
            ques_batch.append(ques_tensor)
            equ_batch.append(equ_tensor)
        ques_batch=self._pad_input_batch(ques_batch,ques_len_batch)
        equ_batch=self._pad_output_batch(equ_batch,equ_len_batch)
        ques_mask_batch=self._get_mask(ques_len_batch)
        equ_mask_batch=self._get_mask(equ_len_batch)
        num_size_batch = [len(num_pos) for num_pos in num_pos_batch]
        num_mask_batch = get_num_mask(num_size_batch, self.dataset.generate_list)

        # to tensor
        ques_tensor_batch = torch.tensor(ques_batch).to(self.device)
        equ_tensor_batch = torch.tensor(equ_batch).to(self.device)
        ques_mask_batch = torch.tensor(ques_mask_batch).to(self.device).bool()
        num_mask_batch = torch.tensor(num_mask_batch).to(self.device).bool()
        ques_len_batch=torch.tensor(ques_len_batch).long()
        equ_mask_batch=torch.tensor(equ_mask_batch).to(self.device).bool()
        return {
            "question": ques_tensor_batch,
            "equation": equ_tensor_batch,
            "ques len":ques_len_batch,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "id": id_batch,
            "num mask": num_mask_batch,
            "ques mask": ques_mask_batch,
            "equ mask":equ_mask_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "num size":num_size_batch,
            "ques_source":ques_source_batch,
            "equ_source":equ_source_batch
        }