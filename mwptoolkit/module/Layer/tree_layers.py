import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.utils.enum_type import SpecialTokens


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class NodeEmbeddingNode:
    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return


class Node():
    def __init__(self, node_value, isleaf=True):
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


class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ""
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level + 1)
            else:
                ret += "\t" * level + str(child) + "\n"
        return ret

    def add_child(self, c):
        if isinstance(c, type(self)):
            c.parent = self
        self.children.append(c)
        self.num_children = self.num_children + 1

    def to_string(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_string() + " )")
            else:
                r_list.append(str(self.children[i]))
        return "".join(r_list)

    def to_list(self, out_idx2symbol):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                #r_list.append(form_manager.get_symbol_idx("("))
                cl = self.children[i].to_list(out_idx2symbol)
                r_list.append(cl)
                # for k in range(len(cl)):
                #     r_list.append(cl[k])
                #r_list.append(form_manager.get_symbol_idx(")"))
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.NON_TOKEN):
                continue
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                continue
            else:
                r_list.append(self.children[i])
        return r_list


class BinaryTree():
    def __init__(self, root_node=None):
        self.root = root_node
        self.rel_quants = []

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

    def equ2tree_(self, equ_list):
        stack = []
        for symbol in equ_list:
            if symbol in [SpecialTokens.EOS_TOKEN, SpecialTokens.PAD_TOKEN]:
                break
            if symbol in ['+', '-', '*', '/', '^', '=', SpecialTokens.BRG_TOKEN, SpecialTokens.OPT_TOKEN]:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                node = Node(symbol, isleaf=True)
                stack.append(node)
        self.root = stack.pop()

    def tree2equ(self, node):
        equation = []
        if node.is_leaf:
            equation.append(node.node_value)
            return equation
        right_equ = self.tree2equ(node.right_node)
        left_equ = self.tree2equ(node.left_node)
        equation = left_equ + right_equ + [node.node_value]
        return equation

    def is_float(self, num_str):
        try:
            float(num_str)
        except:
            return False
        else:
            return True

    def is_equal(self, v1, v2):
        if self.is_float(v1) == False or self.is_float(v2) == False:
            return False
        if abs(float(v1) - float(v2)) < 1e-5:
            return True
        return False

    def lca(self, root, va, vb, parent):
        left = False
        right = False
        if not self.result and root.left_node:
            left = self.lca(root.left_node, va, vb, root)
        if not self.result and root.left_node:
            right = self.lca(root.right_node, va, vb, root)
        mid = False
        if self.is_equal(root.node_value, va) or self.is_equal(root.node_value, vb):
            mid = True
        if not self.result and (left + right + mid) == 2:
            if mid:
                self.result = parent
            else:
                self.result = root
        return left or mid or right

    def is_in_rel_quants(self, value, rel_quants):
        if value in rel_quants:
            return True
        else:
            return False

    def query(self, va, vb):
        if self.root == None:
            return None
        self.result = None
        self.lca(self.root, va, vb, None)
        if self.result:
            return self.result.value
        else:
            return self.result


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
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class ScoreModel(nn.Module):
    def __init__(self, hidden_size):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_size * 3, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, context, token_embeddings):
        # hidden/context: batch_size * hidden_size
        # token_embeddings: batch_size * class_size * hidden_size
        batch_size, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        # (b, c, h)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).view(batch_size, class_size)
        return score


class NodeGenerater(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(NodeGenerater, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_left = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_right = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_left_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_right_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_left(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_left_g(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_right(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_right_g(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_

class NodeEmbeddingLayer(nn.Module):
    def __init__(self, op_nums, embedding_size):
        super(NodeEmbeddingLayer, self).__init__()

        self.embedding_size = embedding_size
        self.op_nums = op_nums

        self.embeddings = nn.Embedding(op_nums, embedding_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)

        return node_embedding, node_embedding, node_label_


class TreeEmbeddingModel(nn.Module):
    def __init__(self, hidden_size, op_set, dropout=0.4):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_size, hidden_size * 2, dropout=dropout, single_layer=True)
        return

    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed

    def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index.cuda()
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for node_label, tree_stack, label_embedding in zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbedding(label_embedding, terminal=False)
            # numbers
            else:
                right_embedding = label_embedding
                # on right tree => merge
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbedding(right_embedding, terminal=True)
            tree_stack.append(tree_node)
        return labels_embedding


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

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class RecursiveNN(nn.Module):
    def __init__(self, emb_size, op_size, op_list):
        super().__init__()
        self.emb_size = emb_size
        self.op_size = op_size
        self.W = nn.Linear(emb_size * 2, emb_size, bias=True)
        self.generate_linear = nn.Linear(emb_size, op_size, bias=True)
        #self.softmax = nn.functional.softmax
        self.classes = op_list

    def forward(self, expression_tree, num_embedding, look_up, out_idx2symbol):
        device = num_embedding.device
        self.out_idx2symbol = out_idx2symbol
        self.leaf_emb(expression_tree, num_embedding, look_up)
        self.nodeProbList = []
        self.labelList = []
        _ = self.traverse(expression_tree)
        if self.nodeProbList != []:
            nodeProb = torch.cat(self.nodeProbList, dim=0).to(device)
            label = torch.tensor(self.labelList).to(device)
        else:
            nodeProb = self.nodeProbList
            label = self.labelList
        return nodeProb, label

    def test(self, expression_tree, num_embedding, look_up, out_idx2symbol):
        self.out_idx2symbol = out_idx2symbol
        self.leaf_emb(expression_tree, num_embedding, look_up)
        self.nodeProbList = []
        self.labelList = []
        _ = self.test_traverse(expression_tree)
        return expression_tree

    def leaf_emb(self, node, num_embed, look_up):
        if node.is_leaf:
            #symbol=self.out_idx2symbol[node.node_value]
            symbol = node.node_value
            if symbol not in look_up:
                node.embedding = num_embed[0]
            else:
                node.embedding = num_embed[look_up.index(symbol)]
        else:
            self.leaf_emb(node.left_node, num_embed, look_up)
            self.leaf_emb(node.right_node, num_embed, look_up)

    def traverse(self, node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.traverse(node.left_node)
            right_vector = self.traverse(node.right_node)

            combined_v = torch.cat((left_vector, right_vector), 1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)

            self.nodeProbList.append(op_prob)
            #node.numclass_probs = proj_probs
            self.labelList.append(self.classes.index(node.node_value))
        return currentNode

    def test_traverse(self, node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.test_traverse(node.left_node)
            right_vector = self.test_traverse(node.right_node)

            combined_v = torch.cat((left_vector, right_vector), 1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)
            op_idx = torch.topk(op_prob, 1, 1)[1]
            node.node_value = self.classes[op_idx]
            #self.nodeProbList.append(op_prob)
            #node.numclass_probs = proj_probs
            #self.labelList.append(self.classes.index(node.node_value))
        return currentNode

    def RecurCell(self, combine_emb):
        node_embedding = torch.tanh(self.W(combine_emb))
        #op=self.softmax(self.generate_linear(node_embedding),dim=1)
        op = self.generate_linear(node_embedding)
        return node_embedding, op


class Dec_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_ratio):
        super(Dec_LSTM, self).__init__()
        #self.opt = opt
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio

        self.i2h = nn.Linear(self.embedding_size + 2 * self.hidden_size, 4 * self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        input_cat = torch.cat((x, parent_h, sibling_state), 1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return cy, hy


class DQN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout_ratio):
        super(DQN, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, embedding_size)
        self.action_pred = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out_1 = self.hidden_layer_1(inputs)
        out_2 = self.hidden_layer_2(out_1)
        pred = self.action_pred(out_1)
        return pred, out_2

    def play_one(self, inputs):
        pred, obv = self.forward(inputs)
        act = pred.topk(1, dim=0)[1]

        return act, obv


class GateNN(nn.Module):
    def __init__(self, hidden_size, input1_size, input2_size=0, dropout=0.4, single_layer=False):
        super(GateNN, self).__init__()
        self.single_layer = single_layer
        self.hidden_l1 = nn.Linear(input1_size + hidden_size, hidden_size)
        self.gate_l1 = nn.Linear(input1_size + hidden_size, hidden_size)
        if not single_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_size + hidden_size, hidden_size)
            self.gate_l2 = nn.Linear(input2_size + hidden_size, hidden_size)
        return

    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.single_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h


class DecomposeModel(nn.Module):
    def __init__(self, hidden_size, dropout, device):
        super(DecomposeModel, self).__init__()
        self.pad_hidden = torch.zeros(hidden_size)
        self.pad_hidden = self.pad_hidden.to(device)

        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_size, hidden_size * 2, 0, dropout=dropout, single_layer=False)
        self.r_decompose = GateNN(hidden_size, hidden_size * 2, hidden_size, dropout=dropout, single_layer=False)
        return

    def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        for node_stack, tree_stack, node_context, label_embedding in zip(node_stacks, tree_stacks, nodes_context, labels_embedding):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden  # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)  # add context and label of parent for right child
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
                # right
                else:
                    node_stack.pop()  # left child, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding  # left tree
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
                    # else finished decode
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        return children_hidden
