import copy
from mwptoolkit.loss.nll_loss import NLLLoss
import torch
from torch import nn

#from mwptoolkit.module.Layer.tree_layers import Node,BinaryTree
from mwptoolkit.module.Layer.tree_layers import RecursiveNN
from mwptoolkit.module.Encoder.rnn_encoder import SelfAttentionRNNEncoder, BasicRNNEncoder
from mwptoolkit.module.Attention.seq_attention import SeqAttention
from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec
from mwptoolkit.utils.data_structure import Node, BinaryTree
from mwptoolkit.utils.enum_type import NumMask, SpecialTokens


class TRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.generate_list = config["generate_list"]
        self.temp_idx2symbol = config["temp_idx2symbol"]
        self.out_idx2symbol = config["out_idx2symbol"]
        self.in_idx2word = config["in_idx2word"]
        self.bidirectional = config["bidirectional"]
        self.hidden_size = config["hidden_size"]
        temp_config["out_idx2symbol"] = temp_config["temp_idx2symbol"]
        temp_config["out_symbol2idx"] = temp_config["temp_symbol2idx"]
        temp_config["symbol_size"] = temp_config["temp_symbol_size"]
        self.seq2seq = RNNEncDec(temp_config)
        self.embedder = BaiscEmbedder(temp_config["vocab_size"], temp_config["embedding_size"], temp_config["dropout_ratio"])
        self.attn_encoder=SelfAttentionRNNEncoder(temp_config["embedding_size"],temp_config["hidden_size"],temp_config["embedding_size"],temp_config["num_layers"],\
                                                    temp_config["rnn_cell_type"],temp_config["dropout_ratio"],temp_config["bidirectional"])
        self.recursivenn = RecursiveNN(temp_config["embedding_size"], temp_config["operator_nums"], temp_config["operator_list"])

        weight2=torch.ones(temp_config["operator_nums"]).to(self.config["device"])
        self.ans_module_loss=NLLLoss(weight2)

    def forward(self, seq, seq_length, num_pos, target=None):
        if target != None:
            return self.generate_t()
        else:
            templates, equations = self.generate_without_t(seq, seq_length, num_pos)
            return templates, equations
    def calculate_loss_seq2seq(self,batch_data):
        #batch_data["question"], batch_data["ques len"], batch_data["template"]
        new_batch_data={}
        new_batch_data["question"]=batch_data["question"]
        new_batch_data["ques len"]=batch_data['ques len']
        new_batch_data['equation']=batch_data['template']
        return self.seq2seq.calculate_loss(new_batch_data)
    def calculate_loss_ans_module(self,batch_data):
        seq=batch_data["question"]
        seq_length=batch_data["ques len"]
        num_pos=batch_data["num pos"]
        template=batch_data["equ_source"]

        device = seq.device
        seq_emb = self.embedder(seq)
        encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)
        batch_size = encoder_output.size(0)
        generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
        generate_num = torch.tensor(generate_num).to(device)
        generate_emb = self.embedder(generate_num)

        batch_prob = []
        batch_target = []
        for b_i in range(batch_size):
            look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            #tree_i=tree[b_i]
            try:
                tree_i = self.template2tree(template[b_i])
            except IndexError:
                continue
            prob, target = self.recursivenn.forward(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
            if prob != []:
                batch_prob.append(prob)
                batch_target.append(target)
        
        self.ans_module_loss.reset()
        for b_i in range(len(target)):
            output=torch.nn.functional.log_softmax(batch_prob[b_i],dim=1)
            self.ans_module_loss.eval_batch(output, target[b_i].view(-1))
        self.ans_module_loss.backward()
        return self.ans_module_loss.get_loss()
    def model_test(self,batch_data):
        new_batch_data={}
        new_batch_data["question"]=batch_data["question"]
        new_batch_data["ques len"]=batch_data['ques len']
        new_batch_data['equation']=batch_data['template']
        outputs,_ = self.seq2seq.model_test(batch_data)
        templates = self.idx2symbol(outputs)
    def generate_t(self):
        raise NotImplementedError("use model.seq2seq_forward() to train seq2seq module, use model.answer_module_forward() to train answer module.")

    def generate_without_t(self, seq, seq_length, num_pos):
        outputs = self.seq2seq(seq, seq_length)
        templates = self.idx2symbol(outputs)
        equations = self.test_ans_module(seq, seq_length, num_pos, templates)
        return outputs, equations

    def seq2seq_forward(self, seq, seq_length, target=None):
        return self.seq2seq(seq, seq_length, target)

    def answer_module_forward(self, seq, seq_length, num_pos, template):
        seq, seq_length, num_pos, template=0
        device = seq.device
        seq_emb = self.embedder(seq)
        encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)
        batch_size = encoder_output.size(0)
        generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
        generate_num = torch.tensor(generate_num).to(device)
        generate_emb = self.embedder(generate_num)
        # tree=[]
        # for temp in template:
        #     tree.append(self.template2tree(temp))
        batch_prob = []
        batch_target = []
        for b_i in range(batch_size):
            look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            #tree_i=tree[b_i]
            try:
                tree_i = self.template2tree(template[b_i])
            except IndexError:
                continue
            prob, target = self.recursivenn.forward(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
            if prob != []:
                batch_prob.append(prob)
                batch_target.append(target)
        return batch_prob, batch_target

    def test_ans_module(self, seq, seq_length, num_pos, template):
        self.wrong=0
        device = seq.device
        seq_emb = self.embedder(seq)
        encoder_output, encoder_hidden = self.attn_encoder(seq_emb, seq_length)

        batch_size = encoder_output.size(0)
        generate_num = [self.in_idx2word.index(SpecialTokens.UNK_TOKEN)] + [self.in_idx2word.index(num) for num in self.generate_list]
        generate_num = torch.tensor(generate_num).to(device)
        generate_emb = self.embedder(generate_num)
        tree = []
        equations = []
        # for temp in template:
        #     tree.append(self.template2tree(temp))
        for b_i in range(batch_size):
            look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + NumMask.number[:len(num_pos[b_i])]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            try:
                tree_i = self.template2tree(template[b_i])
            except:
                equations.append([])
                self.wrong+=1
                continue
            nodes_pred = self.recursivenn.test(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
            tree_i.root = nodes_pred
            equation = self.tree2equation(tree_i)
            equation = self.symbol2idx(equation)
            equations.append(equation)
            # tree_i=self.template2tree(template[b_i])
            # equations.append([])
        return equations

    def template2tree(self, template):
        tree = BinaryTree()
        tree.equ2tree_(template)
        return tree

    def tree2equation(self, tree):
        equation = tree.tree2equ(tree.root)
        return equation

    def idx2symbol(self, output):
        r"""idx to symbol
        tempalte idx to template symbol
        """
        batch_size = output.size(0)
        seq_len = output.size(1)
        symbol_list = []
        for b in range(batch_size):
            symbols = []
            for i in range(seq_len):
                symbols.append(self.temp_idx2symbol[output[b, i]])
            symbol_list.append(symbols)
        return symbol_list

    def symbol2idx(self, symbols):
        r"""symbol to idx
        equation symbol to equation idx
        """
        outputs = []
        for symbol in symbols:
            if symbol not in self.out_idx2symbol:
                idx = self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)
            else:
                idx = self.out_idx2symbol.index(symbol)
            outputs.append(idx)
        return outputs
