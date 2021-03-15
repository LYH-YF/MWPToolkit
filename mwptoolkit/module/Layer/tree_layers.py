import torch
from torch import nn

from mwptoolkit.utils.enum_type import SpecialTokens

class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag

class Node():
    def __init__(self,node_value,isleaf=True):
        self.node_value = node_value
        self.is_leaf = isleaf
        self.embedding = None
        self.left_node = None
        self.right_node = None
    def set_left_node(self, node):
        self.left_node = node
    def set_right_node(self, node):
        self.right_node = node
    
class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

class BinaryTree():
    def __init__(self,root_node=None):
        self.root = root_node
    def equ2tree(self, equ_list, out_idx2symbol, op_list, input_var, emb):
        
        stack = []
        for idx in equ_list:
            if idx == out_idx2symbol.index(SpecialTokens.PAD_TOKEN):
                break
            if idx == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                break
        
            if out_idx2symbol[idx] in op_list:
                node = Node(idx, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                node = Node(idx, isleaf=True)
                position = (input_var == idx).nonzero()
                node.node_embeding = emb[position]
                stack.append(node)
        self.root = stack.pop()
    def equ2tree_(self,equ_list):
        stack=[]
        for symbol in equ_list:
            if symbol in [SpecialTokens.EOS_TOKEN,SpecialTokens.PAD_TOKEN]:
                break
            if symbol in ['+', '-', '*', '/', '^']:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                node = Node(symbol, isleaf=True)
                stack.append(node)
        self.root = stack.pop()
    def tree2equ(self,node):
        equation=[]
        if node.is_leaf:
            equation.append(node.node_value)
            return equation
        right_equ = self.tree2equ(node.right_node)
        left_equ = self.tree2equ(node.left_node)
        equation=left_equ+right_equ+[node.node_value]
        return equation

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings),
                              2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class NodeGenerater(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(NodeGenerater, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_left = nn.Linear(hidden_size * 2 + embedding_size,
                                       hidden_size)
        self.generate_right = nn.Linear(hidden_size * 2 + embedding_size,
                                        hidden_size)
        self.generate_left_g = nn.Linear(hidden_size * 2 + embedding_size,
                                         hidden_size)
        self.generate_right_g = nn.Linear(hidden_size * 2 + embedding_size,
                                          hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(
            self.generate_left(
                torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(
            self.generate_left_g(
                torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(
            self.generate_right(
                torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(
            self.generate_right_g(
                torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class SubTreeMerger(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(SubTreeMerger, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(
            self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(
            self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2),
                                   1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class RecursiveNN(nn.Module):
    def __init__(self,emb_size,op_size):
        super().__init__()
        self.emb_size=emb_size
        self.op_size=op_size
        self.W = nn.Linear(emb_size * 2, emb_size)
        self.generate_linear = nn.Linear(emb_size,op_size)
        self.softmax = nn.functional.softmax
        self.classes=["+","-","*","/","^"]
    
    def forward(self,expression_tree,num_embedding,look_up,out_idx2symbol):
        device=num_embedding.device
        self.out_idx2symbol=out_idx2symbol
        self.leaf_emb(expression_tree,num_embedding,look_up)
        self.nodeProbList=[]
        self.labelList=[]
        _=self.traverse(expression_tree)
        if self.nodeProbList != []:
            nodeProb=torch.cat(self.nodeProbList,dim=0).to(device)
            label=torch.tensor(self.labelList).to(device)
        else:
            nodeProb=self.nodeProbList
            label=self.labelList
        return nodeProb,label
    def test(self,expression_tree,num_embedding,look_up,out_idx2symbol):
        self.out_idx2symbol=out_idx2symbol
        self.leaf_emb(expression_tree,num_embedding,look_up)
        self.nodeProbList=[]
        self.labelList=[]
        _=self.test_traverse(expression_tree)
        return expression_tree
    def leaf_emb(self, node, num_embed, look_up):
        if node.is_leaf:
            #symbol=self.out_idx2symbol[node.node_value]
            symbol=node.node_value
            if symbol not in look_up:
                node.embedding = num_embed[0]
            else:
                node.embedding = num_embed[look_up.index(symbol)]
        else:
            self.leaf_emb(node.left_node, num_embed, look_up)
            self.leaf_emb(node.right_node, num_embed, look_up)
    
    def traverse(self,node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.traverse(node.left_node)
            right_vector = self.traverse(node.right_node)
            
            combined_v = torch.cat((left_vector, right_vector),1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)
            
            self.nodeProbList.append(op_prob)
            #node.numclass_probs = proj_probs 
            self.labelList.append(self.classes.index(node.node_value))
        return currentNode
    def test_traverse(self,node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.test_traverse(node.left_node)
            right_vector = self.test_traverse(node.right_node)
            
            combined_v = torch.cat((left_vector, right_vector),1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)
            op_prob = self.softmax(op_prob,dim=1)
            op_idx = torch.topk(op_prob,1,1)[1]
            node.node_value = self.classes[op_idx]
            #self.nodeProbList.append(op_prob)
            #node.numclass_probs = proj_probs 
            #self.labelList.append(self.classes.index(node.node_value))
        return currentNode
    def RecurCell(self,combine_emb):
        node_embedding=self.W(combine_emb)
        #op=self.softmax(self.generate_linear(node_embedding))
        op=self.generate_linear(node_embedding)
        return node_embedding,op