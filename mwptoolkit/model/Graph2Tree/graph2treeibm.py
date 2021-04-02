import random
import torch
from torch import nn

from mwptoolkit.module.Encoder.graph_based_encoder import GraphEncoder
from mwptoolkit.module.Attention.separate_attention import SeparateAttention
from mwptoolkit.module.Decoder.tree_decoder import RNNBasedTreeDecoder
from mwptoolkit.module.Layer.tree_layers import Tree

class Graph2TreeIBM(nn.Module):
    def __init__(self,config):
        super(Graph2TreeIBM,self).__init__()
        self.hidden_size=config["hidden_size"]
        self.teacher_force_ratio=config["teacher_force_ratio"]
        self.max_length=config["max_output_len"]

        self.encoder=GraphEncoder(config["vocab_size"],config["embedding_size"],config["hidden_size"],\
                                    config["sample_size"],config["sample_layer"],config["bidirectional"],config["encoder_dropout_ratio"])
        self.attention=SeparateAttention(config["hidden_size"],config["symbol_size"],config["attention_dropout_ratio"])
        self.decoder=RNNBasedTreeDecoder(config["vocab_size"],config["embedding_size"],config["hidden_size"],config["decoder_dropout_ratio"])

    def forward(self,seq,seq_length,group_nums,target=None):
        
        enc_max_len = seq_length.max()
        batch_size = seq_length.size(0)


        enc_outputs = torch.zeros((batch_size, enc_max_len, self.hidden_size), requires_grad=True)
        

        # fw_adj_info = torch.tensor(enc_batch['g_fw_adj'])
        # bw_adj_info = torch.tensor(enc_batch['g_bw_adj'])
        # feature_info = torch.tensor(enc_batch['g_ids_features'])
        # batch_nodes = torch.tensor(enc_batch['g_nodes'])
        fw_adj_info,bw_adj_info,feature_info,batch_nodes=self.build_graph()

        node_embedding, graph_embedding, structural_info = self.encoder(fw_adj_info,bw_adj_info,feature_info,batch_nodes)


        enc_outputs = node_embedding

        graph_cell_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
        graph_hidden_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)

        graph_cell_state = graph_embedding
        graph_hidden_state = graph_embedding

        dec_s = {}
        for i in range(self.max_length + 1):
            dec_s[i] = {}
            for j in range(self.max_length + 1):
                dec_s[i][j] = {}

        loss = 0
        cur_index = 1

        dec_batch, queue_tree, max_index = self.get_dec_batch(target, form_manager)
        while (cur_index <= max_index):
            for j in range(1, 3):
                dec_s[cur_index][0][j] = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True).to(device)
                # if using_gpu:
                #     dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

        sibling_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True).to(device)
        # if using_gpu:
        #         sibling_state = sibling_state.cuda()

        if cur_index == 1:
            for i in range(batch_size):
                dec_s[1][0][1][i, :] = graph_cell_state[i]
                dec_s[1][0][2][i, :] = graph_hidden_state[i]

        else:
            for i in range(1, batch_size+1):
                if (cur_index <= len(queue_tree[i])):
                    par_index = queue_tree[i][cur_index - 1]["parent"]
                    child_index = queue_tree[i][cur_index - 1]["child_index"]
                    
                    dec_s[cur_index][0][1][i-1,:] = \
                        dec_s[par_index][child_index][1][i-1,:]
                    dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]

                flag_sibling = False
                for q_index in range(len(queue_tree[i])):
                    if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state[i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1,:]
                
        parent_h = dec_s[cur_index][0][2]
        for i in range(dec_batch[cur_index].size(1) - 1):
            teacher_force = random.random() < self.teacher_force_ratio
            if teacher_force != True and i > 0:
                input_word = pred.argmax(1)
            else:
                input_word = dec_batch[cur_index][:, i]

            dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = self.decoder(input_word, dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, sibling_state)
            pred = self.attention(enc_outputs, dec_s[cur_index][i+1][2], structural_info)
            #loss += criterion(pred, dec_batch[cur_index][:,i+1])
        cur_index = cur_index + 1
    
    def build_graph(self,group_nums,seq_length):
        max_length=seq_length.max()
        batch_size=len(seq_length)
        for b_i in range(batch_size):
            x=torch.zeros((max_length,max_length))
            for idx in range(seq_length[b_i]):
                x[idx,idx]=1
            fw_adj_info=torch.clone(x)
            bw_adj_info=torch.clone(x)
            for idx in group_nums[b_i]:
                fw_adj_info[idx[0],idx[1]]=1
                bw_adj_info[idx[1],idx[0]]=1
        fw_adj_info = []
        bw_adj_info = []
        feature_info = []
        batch_nodes = []
        fw_adj_info = torch.tensor(fw_adj_info)
        bw_adj_info = torch.tensor(bw_adj_info)
        feature_info = torch.tensor(feature_info)
        batch_nodes = torch.tensor(batch_nodes)

        return fw_adj_info,bw_adj_info,feature_info,batch_nodes
    
    def get_dec_batch(self,dec_tree_batch, form_manager, batch_size):
        queue_tree = {}
        for i in range(1, batch_size+1):
            queue_tree[i] = []
            queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})

        cur_index, max_index = 1,1
        dec_batch = {}
        # max_index: the max number of sequence decoder in one batch
        while (cur_index <= max_index):
            max_w_len = -1
            batch_w_list = []
            for i in range(1, batch_size+1):
                w_list = []
                if (cur_index <= len(queue_tree[i])):
                    t = queue_tree[i][cur_index - 1]["tree"]

                    for ic in range (t.num_children):
                        if isinstance(t.children[ic], Tree):
                            w_list.append(4)
                            queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic + 1})
                        else:
                            w_list.append(t.children[ic])
                    if len(queue_tree[i]) > max_index:
                        max_index = len(queue_tree[i])
                if len(w_list) > max_w_len:
                    max_w_len = len(w_list)
                batch_w_list.append(w_list)
            dec_batch[cur_index] = torch.zeros((batch_size, max_w_len + 2), dtype=torch.long)
            for i in range(batch_size):
                w_list = batch_w_list[i]
                if len(w_list) > 0:
                    for j in range(len(w_list)):
                        dec_batch[cur_index][i][j+1] = w_list[j]
                    # add <S>, <E>
                    if cur_index == 1:
                        dec_batch[cur_index][i][0] = 1
                    else:
                        dec_batch[cur_index][i][0] = form_manager.get_symbol_idx('(')
                    dec_batch[cur_index][i][len(w_list) + 1] = 2

            # if using_gpu:
            #   dec_batch[cur_index] = dec_batch[cur_index].cuda()
            cur_index += 1

        return dec_batch, queue_tree, max_index