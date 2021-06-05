import numpy as np
import torch

from mwptoolkit.data.dataloader.template_dataloader import TemplateDataLoader
from mwptoolkit.data.dataset.dataset_multiead import DatasetMultiEAD
from mwptoolkit.utils.utils import str2float

class DataLoaderMultiEAD(TemplateDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.dataset=DatasetMultiEAD(config)
    def load_batch(self, batch):
        input1_batch=[]
        input2_batch=[]
        output1_batch=[]
        output2_batch=[]

        input1_length_batch=[]
        input2_length_batch=[]
        output1_length_batch=[]
        output2_length_batch=[]

        num_list_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_order_batch = []
        parse_graph_batch = []

        id_batch=[]
        ans_batch=[]
        for data in batch:
            input1_tensor=[]
            input2_tensor=[]
            # question word to index
            sentence = data['question']
            pos = data['pos']
            prefix_equ = data['prefix']
            postfix_equ = data['postfix']
            if self.add_sos:
                input1_tensor.append(self.dataset.in_word2idx_1["<SOS>"])
            input1_tensor += self._word2idx_1(sentence)
            if self.add_eos:
                input1_tensor.append(self.dataset.in_word2idx_1["<EOS>"])
            
            input2_tensor+=self._word2idx_2(pos)
            # equation symbol to index
            prefix_equ_tensor = self._equ_symbol2idx_1(prefix_equ)
            postfix_equ_tensor = self._equ_symbol2idx_2(postfix_equ)

            postfix_equ_tensor.append(self.dataset.out_idx2symbol_2.index('<EOS>'))

            input1_length_batch.append(len(input1_tensor))
            output1_length_batch.append(len(prefix_equ_tensor))
            output2_length_batch.append(len(postfix_equ_tensor))
            
            input1_batch.append(input1_tensor)
            input2_batch.append(input2_tensor)
            output1_batch.append(prefix_equ_tensor)
            output2_batch.append(postfix_equ_tensor)

            num_list=[str2float(n) for n in data['num list']]
            num_list_batch.append(num_list)
            num_order=self.num_order_processed(num_list)
            num_order_batch.append(num_order)
            num_stack_batch.append(self._build_num_stack(prefix_equ, data["number list"]))
            parse_graph_batch.append(self.get_parse_graph_batch(data['parse tree']))

            if self.add_sos:
                num_pos = [pos + 1 for pos in data["number position"]]  # pos plus one because of adding <SOS> at the head of sentence
            else:
                num_pos = [pos for pos in data["number position"]]
            num_pos_batch.append(num_pos)
            # quantity count
            # question id and answer
            id_batch.append(data["id"])
            ans_batch.append(data["ans"])
        num_size_batch = [len(num_pos) for num_pos in num_pos_batch]

        return {
            "input1": input1_batch,
            "input2": input2_batch,
            "output1": output1_batch,
            "output2": output2_batch,
            "input1 len": input1_length_batch,
            "output1 len": output1_length_batch,
            "output2 len":output2_length_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "id": id_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "num size": num_size_batch,
            "graph":parse_graph_batch
        }
    def _word2idx_1(self, sentence):
        sentence_idx = []
        for word in sentence:
            try:
                idx = self.dataset.in_word2idx_1[word]
            except:
                idx = self.in_unk_token
            sentence_idx.append(idx)
        return sentence_idx
    def _word2idx_2(self, pos):
        pos_idx = []
        for word in pos:
            try:
                idx = self.dataset.in_word2idx_2[word]
            except:
                idx = self.in_unk_token
            pos_idx.append(idx)
        return pos_idx
    def _equ_symbol2idx_1(self, equation):
        equ_idx = []
        for word in equation:
            try:
                idx = self.dataset.out_symbol2idx_1[word]
            except:
                idx = self.out_unk_token
            equ_idx.append(idx)
        return equ_idx
    
    def _equ_symbol2idx_2(self, equation):
        equ_idx = []
        for word in equation:
            try:
                idx = self.dataset.out_symbol2idx_2[word]
            except:
                idx = self.out_unk_token
            equ_idx.append(idx)
        return equ_idx

    def _build_num_stack(self, equation, num_list):
        num_stack = []
        for word in equation:
            temp_num = []
            flag_not = True
            if word not in self.dataset.out_idx2symbol_1:
                flag_not = False
                if "NUM" in word:
                    temp_num.append(int(word[4:]))
                for i, j in enumerate(num_list):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(num_list))])
        num_stack.reverse()
        return num_stack
    
    def num_order_processed(self,num_list):
        num_order = []
        num_array = np.asarray(num_list)
        for num in num_array:
            num_order.append(sum(num>num_array)+1)
        
        return num_order
    def get_parse_graph_batch(input_length, parse_tree_batch):
        batch_graph = []
        max_len = max(input_length)
        for i in range(len(input_length)):
            parse_tree = parse_tree_batch[i]
            diag_ele = [1] * input_length[i] + [0] * (max_len - input_length[i])
            #graph1 = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
            graph1=torch.diag(torch.tensor([1]*max_len))+torch.diag(torch.tensor(diag_ele[1:]),1)+torch.diag(torch.tensor(diag_ele[1:]),-1)
            graph2 = graph1.clone()
            graph3 = graph1.clone()
            for j in range(len(parse_tree)):
                if parse_tree[j] != -1:
                    graph1[j, parse_tree[j]] = 1
                    graph2[parse_tree[j], j] = 1
                    graph3[j, parse_tree[j]] = 1
                    graph3[parse_tree[j], j] = 1
            graph = [graph1.tolist(), graph2.tolist(), graph3.tolist()]
            batch_graph.append(graph)
        batch_graph = torch.tensor(batch_graph)
        return batch_graph
