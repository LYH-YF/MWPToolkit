import copy
import torch
from torch import nn

from mwptoolkit.module.Layer.tree_layers import Node,BinaryTree
from mwptoolkit.module.Layer.tree_layers import RecursiveNN
from mwptoolkit.module.Encoder.rnn_encoder import SelfAttentionRNNEncoder
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens

class TRNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        temp_config=copy.deepcopy(config)
        self.generate_list=config["generate_list"]
        self.temp_idx2symbol=config["temp_idx2symbol"]
        self.out_idx2symbol=config["out_idx2symbol"]
        temp_config["out_idx2symbol"]=temp_config["temp_idx2symbol"]
        temp_config["out_symbol2idx"]=temp_config["temp_symbol2idx"]
        temp_config["symbol_size"]=temp_config["temp_symbol_size"]
        self.seq2seq = RNNEncDec(temp_config)
        self.embedder=BaiscEmbedder(temp_config["vocab_size"],temp_config["embedding_size"],temp_config["dropout_ratio"])
        self.attn_encoder=SelfAttentionRNNEncoder(temp_config["embedding_size"],temp_config["hidden_size"],temp_config["num_layers"],\
                                                    temp_config["rnn_cell_type"],temp_config["dropout_ratio"],temp_config["bidirectional"])
        self.recursivenn=RecursiveNN(temp_config["embedding_size"],temp_config["operator_nums"])
    
    def forward(self,seq,seq_length,target=None):
        if target != None:
            self.generate_t()
        else:
            self.generate_without_t()
    def generate_t(self):
        pass
    def generate_without_t(self):
        pass
    def seq2seq_forward(self,seq,seq_length,target=None):
        return self.seq2seq(seq,seq_length,target)
    def recursivenn_forward(self,seq,seq_length,template=None):
        tree=self.template2tree(template,seq,)
    
    def answer_module_forward(self,seq,seq_length,num_pos, template):
        device=seq.device
        seq_emb=self.embedder(seq)
        encoder_output,encoder_hidden=self.attn_encoder(seq_emb,seq_length)
        batch_size=encoder_output.size(0)
        generate_num=[self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)]+[self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num=torch.tensor(generate_num).to(device)
        generate_emb=self.embedder(generate_num)
        tree=[]
        for temp in template:
            tree.append(self.template2tree(temp))
        batch_prob=[]
        batch_target=[]
        for b_i in range(batch_size):
            look_up = [SpecialTokens.UNK_TOKEN]+self.generate_list + NumMask.number[:len(num_pos[b_i])]
            x=encoder_output[b_i,num_pos[b_i]]
            num_embedding=torch.cat([generate_emb,encoder_output[b_i,num_pos[b_i]]],dim=0)
            tree_i=tree[b_i]
            prob,target=self.recursivenn.forward(tree_i.root,num_embedding,look_up,self.out_idx2symbol)
            if prob != []:
                batch_prob.append(prob)
                batch_target.append(target)
        return batch_prob,batch_target
    def test_ans_module(self,seq,seq_length,num_pos, template):
        device=seq.device
        seq_emb=self.embedder(seq)
        encoder_output,encoder_hidden=self.attn_encoder(seq_emb,seq_length)
        batch_size=encoder_output.size(0)
        generate_num=[self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num=torch.tensor(generate_num).to(device)
        generate_emb=self.embedder(generate_num)
        tree=[]
        equations=[]
        for temp in template:
            tree.append(self.template2tree(temp))
        for b_i in range(batch_size):
            look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
            x=encoder_output[b_i,num_pos[b_i]]
            num_embedding=torch.cat([generate_emb,encoder_output[b_i,num_pos[b_i]]],dim=0)
            tree_i=tree[b_i]
            nodes_pred=self.recursivenn.test(tree_i.root,num_embedding,look_up,self.out_idx2symbol)
            tree_i.root=nodes_pred
            equation=self.tree2equation(tree_i)
            equation=self.symbol2idx(equation)
            equations.append(equation)
        return equations
    def template2tree(self,template):
        tree=BinaryTree()
        tree.equ2tree_(template)
        return tree
    def tree2equation(self,tree):
        equation=tree.tree2equ(tree.root)
        return equation
    def idx2symbol(self,output):
        batch_size=output.size(0)
        seq_len=output.size(1)
        symbol_list=[]
        for b in range(batch_size):
            symbols=[]
            for i in range(seq_len):
                symbols.append(self.temp_idx2symbol[output[b,i]])
            symbol_list.append(symbols)
        return symbol_list
    def symbol2idx(self,symbols):
        outputs=[]
        for symbol in symbols:
            idx=self.out_idx2symbol.index(symbol)
            outputs.append(idx)
        return outputs


