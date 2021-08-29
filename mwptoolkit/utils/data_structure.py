# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:14:55
# @File: data_structure.py


from mwptoolkit.utils.enum_type import SpecialTokens, NumMask


class Node():
    """node
    """
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


class AbstractTree():
    def __init__(self):
        self.root = None

    def equ2tree():
        raise NotImplementedError

    def tree2equ():
        raise NotImplementedError


class BinaryTree(AbstractTree):
    """binary tree
    """
    def __init__(self, root_node=None):
        super().__init__()
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
        if len(stack)>1:
            raise IndexError
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

class PrefixTree(BinaryTree):
    def __init__(self, root_node):
        super().__init__(root_node=root_node)

    def prefix2tree(self,equ_list):
        stack = []
        for symbol in equ_list[::-1]:
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


class GoldTree(AbstractTree):
    def __init__(self, root_node=None, gold_ans=None):
        super().__init__()
        self.root = root_node
        self.gold_ans = gold_ans

    def equ2tree(self, equ_list, out_idx2symbol, op_list, num_list, ans):
        stack = []
        for idx in equ_list:
            if idx == out_idx2symbol.index(SpecialTokens.PAD_TOKEN):
                break
            if idx == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                break
            symbol = out_idx2symbol[idx]
            if symbol in op_list:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                if symbol in NumMask.number:
                    i = NumMask.number.index(symbol)
                    value = num_list[i]
                    node = Node(value, isleaf=True)
                elif symbol == SpecialTokens.UNK_TOKEN:
                    node = Node('-inf', isleaf=True)
                else:
                    node = Node(symbol, isleaf=True)
                stack.append(node)
        self.root = stack.pop()
        self.gold_ans = ans

    def is_float(self, num_str, num_list):
        if num_str in num_list:
            return True
        else:
            return False

    def is_equal(self, v1, v2):
        if v1 == v2:
            return True
        else:
            return False

    def lca(self, root, va, vb, parent):
        left = False
        right = False
        if not self.result and root.left_node:
            left = self.lca(root.left_node, va, vb, root)
        if not self.result and root.right_node:
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
            return self.result.node_value
        else:
            return self.result


class DependencyNode():
    def __init__(self, node_value, position, relation, is_leaf=True):
        self.node_value = node_value
        self.position = position
        self.relation = relation
        self.embedding = None
        self.left_nodes = []
        self.right_nodes = []
        self.is_leaf = is_leaf

    def add_left_node(self, node):
        self.left_nodes.append(node)

    def add_right_node(self, node):
        self.right_nodes.append(node)


class DependencyTree():
    def __init__(self, root_node=None):
        self.root = root_node

    def sentence2tree(self, sentence, dependency_info):
        r'''
        dependency info [relation,child,father]
        '''
        node_dict = {}
        for r, c, f in dependency_info:
            if f in node_dict:
                node_dict[f].append((r, c))
            else:
                node_dict[f] = [(r, c)]
        relation, root_idx = node_dict[-1][0]
        child_list = node_dict.get(root_idx, [])
        if child_list:
            node = DependencyNode(sentence[root_idx], root_idx, relation, is_leaf=False)
            left_list, right_list = self._build_sub_node(root_idx, child_list, node_dict, sentence)
            for child in left_list:
                node.add_left_node(child)
            for child in right_list:
                node.add_right_node(child)
        else:
            node = DependencyNode(sentence[root_idx], root_idx, relation)
        self.root = node

    def _build_sub_node(self, father_idx, child_list, node_dict, sentence):
        left_list = []
        right_list = []
        for relation, child_idx in child_list:
            sub_child_list = node_dict.get(child_idx, [])
            if sub_child_list:
                child_node = DependencyNode(sentence[child_idx], child_idx, relation, is_leaf=False)
                sub_left_list, sub_right_list = self._build_sub_node(child_idx, sub_child_list, node_dict, sentence)
                for node in sub_left_list:
                    child_node.add_left_node(node)
                for node in sub_right_list:
                    child_node.add_right_node(node)
            else:
                child_node = DependencyNode(sentence[child_idx], child_idx, relation)
            if child_idx < father_idx:
                left_list.append(child_node)
            else:
                right_list.append(child_node)
        return left_list, right_list


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
                cl = self.children[i].to_list(out_idx2symbol)
                r_list.append(cl)
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.NON_TOKEN):
                continue
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                continue
            else:
                r_list.append(self.children[i])
        return r_list

