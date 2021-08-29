from mwptoolkit.utils.enum_type import SpecialTokens
import random
import torch
from torch import nn

from mwptoolkit.module.Encoder.graph_based_encoder import GraphEncoder
from mwptoolkit.module.Attention.separate_attention import SeparateAttention
from mwptoolkit.module.Decoder.tree_decoder import RNNBasedTreeDecoder
from mwptoolkit.utils.data_structure import Tree


class Graph2TreeIBM(nn.Module):
    def __init__(self, config):
        super(Graph2TreeIBM, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.teacher_force_ratio = config["teacher_force_ratio"]
        self.max_length = config["max_output_len"]
        self.out_idx2symbol=config['out_idx2symbol']

        self.encoder=GraphEncoder(config["vocab_size"],config["embedding_size"],config["hidden_size"],\
                                    config["sample_size"],config["sample_layer"],config["bidirectional"],config["encoder_dropout_ratio"])
        self.attention = SeparateAttention(config["hidden_size"], config["symbol_size"], config["attention_dropout_ratio"])
        self.decoder = RNNBasedTreeDecoder(config["vocab_size"], config["embedding_size"], config["hidden_size"], config["decoder_dropout_ratio"])

    def forward(self, seq, seq_length, group_nums, target=None):

        enc_max_len = seq_length.max()
        batch_size = seq_length.size(0)
        device=seq.device

        enc_outputs = torch.zeros((batch_size, enc_max_len, self.hidden_size), requires_grad=True)

        fw_adj_info, bw_adj_info, nodes = self.build_graph(group_nums, seq_length)
        fw_adj_info = fw_adj_info.to(device)
        bw_adj_info = bw_adj_info.to(device)
        nodes = nodes.to(device)

        node_embedding, graph_embedding, structural_info = self.encoder(fw_adj_info, bw_adj_info, seq, nodes)

        #enc_outputs = node_embedding

        # graph_cell_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
        # graph_hidden_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)

        # graph_cell_state = graph_embedding
        # graph_hidden_state = graph_embedding
        if target != None:
            predict,label=self.generate_t(node_embedding, graph_embedding, structural_info, target)
            return predict,label
        else:
            outputs=self.generate_without_t(node_embedding, graph_embedding, structural_info)
            return outputs

    def generate_t(self,node_embedding,graph_embedding,structural_info,target):
        device=node_embedding.device
        batch_size=node_embedding.size(0)

        enc_outputs = node_embedding
        graph_cell_state = graph_embedding
        graph_hidden_state = graph_embedding
        

        tree_batch=[]
        for tar_equ in target:
            tree_batch.append(self.equ2tree(tar_equ))
        

        dec_batch, queue_tree, max_index = self.get_dec_batch(tree_batch, batch_size)
        predict=[]
        label=[]
        dec_s = {}
        for i in range(1, self.max_length + 1):
            dec_s[i] = {}
            for j in range(self.max_length + 1):
                dec_s[i][j] = {}
        cur_index = 1
        while (cur_index <= max_index):
            for j in range(1, 3):
                dec_s[cur_index][0][j] = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True).to(device)
                

            sibling_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True).to(device)
            

            if cur_index == 1:
                for b_i in range(batch_size):
                    dec_s[1][0][1][b_i, :] = graph_cell_state[b_i]
                    dec_s[1][0][2][b_i, :] = graph_hidden_state[b_i]

            else:
                for b_i in range(1, batch_size + 1):
                    if (cur_index <= len(queue_tree[b_i])):
                        par_index = queue_tree[b_i][cur_index - 1]["parent"]
                        child_index = queue_tree[b_i][cur_index - 1]["child_index"]

                        dec_s[cur_index][0][1][b_i-1,:] = \
                            dec_s[par_index][child_index][1][b_i-1,:]
                        dec_s[cur_index][0][2][b_i - 1, :] = dec_s[par_index][child_index][2][b_i - 1, :]

                    flag_sibling = False
                    for q_index in range(len(queue_tree[b_i])):
                        if (cur_index <= len(queue_tree[b_i])) and (q_index < cur_index - 1) and (queue_tree[b_i][q_index]["parent"] == queue_tree[b_i][cur_index - 1]["parent"]) and (
                                queue_tree[b_i][q_index]["child_index"] < queue_tree[b_i][cur_index - 1]["child_index"]):
                            flag_sibling = True
                            sibling_index = q_index
                    if flag_sibling:
                        sibling_state[b_i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][b_i - 1, :]

            parent_h = dec_s[cur_index][0][2]
            for i in range(dec_batch[cur_index].size(1) - 1):
                teacher_force = random.random() < self.teacher_force_ratio
                if teacher_force != True and i > 0:
                    input_word = pred.argmax(1)
                else:
                    input_word = dec_batch[cur_index][:, i].to(device)
                #if cur_index==3 and 
                dec_s[cur_index][i + 1][1], dec_s[cur_index][i + 1][2] = self.decoder(input_word, dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, sibling_state)
                pred = self.attention(enc_outputs, dec_s[cur_index][i + 1][2], structural_info)
                #loss += criterion(pred, dec_batch[cur_index][:,i+1])
                predict.append(pred)
                label.append(dec_batch[cur_index][:,i+1])
            cur_index = cur_index + 1
        predict=torch.stack(predict,dim=1).to(device)
        label=torch.stack(label,dim=1).to(device)
        #label=label.view(batch_size,-1)
        predict=predict.view(-1,predict.size(2))
        label=label.view(-1,label.size(1))
        return predict,label
    
    def generate_without_t(self,node_embedding,graph_embedding,structural_info):
        batch_size=node_embedding.size(0)
        device=node_embedding.device
        enc_outputs = node_embedding
        prev_c = graph_embedding
        prev_h = graph_embedding
        outputs=[]
        for b_i in range(batch_size):
            queue_decode = []
            queue_decode.append({"s": (prev_c[b_i].unsqueeze(0), prev_h[b_i].unsqueeze(0)), "parent":0, "child_index":1, "t": Tree()})
            head = 1
            while head <= len(queue_decode) and head <=100:
                s = queue_decode[head-1]["s"]
                parent_h = s[1]
                t = queue_decode[head-1]["t"]

                sibling_state = torch.zeros((1, self.encoder.hidden_size), dtype=torch.float, requires_grad=False).to(device)

                flag_sibling = False
                for q_index in range(len(queue_decode)):
                    if (head <= len(queue_decode)) and (q_index < head - 1) and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state = queue_decode[sibling_index]["s"][1]

                if head == 1:
                    prev_word = torch.tensor([self.out_idx2symbol.index(SpecialTokens.SOS_TOKEN)], dtype=torch.long).to(device)
                else:
                    prev_word = torch.tensor([self.out_idx2symbol.index(SpecialTokens.NON_TOKEN)], dtype=torch.long).to(device)
                
                i_child = 1
                while True:
                    curr_c, curr_h = self.decoder(prev_word, s[0], s[1], parent_h, sibling_state)
                    prediction = self.attention(enc_outputs[b_i].unsqueeze(0), curr_h, structural_info[b_i].unsqueeze(0))
        
                    s = (curr_c, curr_h)
                    _, _prev_word = prediction.max(1)
                    prev_word = _prev_word

                    if int(prev_word[0]) == self.out_idx2symbol.index(SpecialTokens.EOS_TOKEN) or t.num_children >= self.max_length:
                        break
                    elif int(prev_word[0]) == self.out_idx2symbol.index(SpecialTokens.NON_TOKEN):
                        queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                        t.add_child(int(prev_word[0]))
                    else:
                        t.add_child(int(prev_word[0]))
                    i_child = i_child + 1
                head = head + 1
            for i in range(len(queue_decode)-1, 0, -1):
                cur = queue_decode[i]
                queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]
            output=queue_decode[0]["t"].to_list(self.out_idx2symbol)
            outputs.append(output)
        return outputs
    
    def build_graph(self, group_nums, seq_length):
        max_length = seq_length.max()
        batch_size = len(seq_length)
        max_degree=6
        fw_adj_info_batch=[]
        bw_adj_info_batch=[]
        slide=0
        for b_i in range(batch_size):
            x=torch.zeros((max_length,max_degree)).long()
            fw_adj_info = torch.clone(x)
            bw_adj_info = torch.clone(x)
            fw_idx=torch.zeros(max_length).long()
            bw_idx=torch.zeros(max_length).long()
            for idx in group_nums[b_i]:
                if fw_idx[idx[0]] < max_degree:
                    fw_adj_info[idx[0], fw_idx[idx[0]]] = idx[1]+slide
                    fw_idx[idx[0]]+=1
                if bw_idx[idx[1]] < max_degree:
                    bw_adj_info[idx[1], bw_idx[idx[1]]] = idx[0]+slide
                    bw_idx[idx[1]]+=1
            for row_idx,col_idx in enumerate(fw_idx):
                for idx_slide in range(max_degree-col_idx):
                    fw_adj_info[row_idx,col_idx+idx_slide]=max_length-1+slide
            for row_idx,col_idx in enumerate(bw_idx):
                for idx_slide in range(max_degree-col_idx):
                    bw_adj_info[row_idx,col_idx+idx_slide]=max_length-1+slide
            #fw_adj_info+=slide
            #bw_adj_info+=slide

            fw_adj_info_batch.append(fw_adj_info)
            bw_adj_info_batch.append(bw_adj_info)
            slide+=max_length
        fw_adj_info_batch=torch.cat(fw_adj_info_batch,dim=0)
        bw_adj_info_batch=torch.cat(bw_adj_info_batch,dim=0)
        nodes_batch=torch.arange(0,fw_adj_info_batch.size(0)).view(batch_size,max_length)
        # for b_i in range(batch_size):
        #     x = torch.zeros((max_length, max_length))
        #     for idx in range(seq_length[b_i]):
        #         x[idx, idx] = 1
        #         if idx == 0:
        #             x[idx, idx + 1] = 1
        #             continue
        #         if idx == seq_length[b_i]-1:
        #             x[idx - 1, idx] = 1
        #             continue
        #         x[idx - 1, idx] = 1
        #         x[idx, idx + 1] = 1
        #     fw_adj_info = torch.clone(x)
        #     bw_adj_info = torch.clone(x)
        #     for idx in group_nums[b_i]:
        #         fw_adj_info[idx[0], idx[1]] = 1
        #         bw_adj_info[idx[1], idx[0]] = 1
        

        return fw_adj_info_batch, bw_adj_info_batch,nodes_batch

    def get_dec_batch(self, dec_tree_batch, batch_size):
        queue_tree = {}
        for i in range(1, batch_size + 1):
            queue_tree[i] = []
            queue_tree[i].append({"tree": dec_tree_batch[i - 1], "parent": 0, "child_index": 1})

        cur_index, max_index = 1, 1
        dec_batch = {}
        # max_index: the max number of sequence decoder in one batch
        while (cur_index <= max_index):
            max_w_len = -1
            batch_w_list = []
            for i in range(1, batch_size + 1):
                w_list = []
                counts = 0
                if (cur_index <= len(queue_tree[i])):
                    t = queue_tree[i][cur_index - 1]["tree"]

                    for ic in range(t.num_children):
                        if isinstance(t.children[ic], Tree):
                            #w_list.append(4)
                            queue_tree[i].append({"tree": t.children[ic], "parent": cur_index, "child_index": ic + 1 - counts})
                            counts+=1
                        else:
                            w_list.append(t.children[ic])
                    if len(queue_tree[i]) > max_index:
                        max_index = len(queue_tree[i])
                if len(w_list) > max_w_len:
                    max_w_len = len(w_list)
                batch_w_list.append(w_list)
            # if cur_index == 1:
            #     dec_batch[cur_index] = torch.zeros((batch_size, max_w_len + 1), dtype=torch.long)
            #     for i in range(batch_size):
            #         w_list = batch_w_list[i]
            #         if len(w_list) > 0:
            #             for j in range(len(w_list)):
            #                 dec_batch[cur_index][i][j + 1] = w_list[j]
            #         # add <SOS>
            #         dec_batch[cur_index][i][0] = self.out_idx2symbol.index(SpecialTokens.SOS_TOKEN)
            # else:
            #     dec_batch[cur_index] = torch.zeros((batch_size, max_w_len), dtype=torch.long)
            #     for i in range(batch_size):
            #         w_list = batch_w_list[i]
            #         if len(w_list) > 0:
            #             for j in range(len(w_list)):
            #                 dec_batch[cur_index][i][j] = w_list[j]
                    
            dec_batch[cur_index] = torch.zeros((batch_size, max_w_len + 1), dtype=torch.long)
            for i in range(batch_size):
                w_list = batch_w_list[i]
                if len(w_list) > 0:
                    for j in range(len(w_list)):
                        dec_batch[cur_index][i][j + 1] = w_list[j]
                # add <SOS> or <NON>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = self.out_idx2symbol.index(SpecialTokens.SOS_TOKEN)
                else:
                    dec_batch[cur_index][i][0] = self.out_idx2symbol.index(SpecialTokens.NON_TOKEN)
            cur_index += 1

        return dec_batch, queue_tree, max_index
    
    def equ2tree(self,equ):
        t=Tree()
        for symbol in equ:
            if isinstance(symbol,list):
                sub_tree=self.equ2tree(symbol)
                t.add_child(self.out_idx2symbol.index(SpecialTokens.NON_TOKEN))
                t.add_child(sub_tree)
            else:
                t.add_child(symbol)
        t.add_child(self.out_idx2symbol.index(SpecialTokens.EOS_TOKEN))
        return t
    # def equ2tree(self,infix_equ):
    #     t=Tree()
    #     st=[]
    #     level=0
    #     for symbol in infix_equ:
    #         if symbol in ['(','[']:
    #             level+=1
    #             st.append(symbol)
    #         elif symbol in [')',']']:
    #             level-=1
    #             st.append(symbol)
    #             if level==0:
    #                 sub_t=self.equ2tree(st[1:-1])
    #             t.add_child(sub_t)
    #         else:
    #             if level!=0:
    #                 if symbol not in self.out_idx2symbol:
    #                     idx=self.out_idx2symbol.index('<UNK>')
    #                 else:
    #                     idx=self.out_idx2symbol.index(symbol)
    #                 t.add_child(idx)
    #             else:
    #                 st.append(symbol)
    #     return t

            


