from mwptoolkit.utils.enum_type import SpecialTokens,NumMask

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


class AbstractTree():
    def __init__(self):
        self.root=None
    def equ2tree():
        raise NotImplementedError
    def tree2equ():
        raise NotImplementedError

class BinaryTree(AbstractTree):
    def __init__(self,root_node=None):
        super().__init__()
        self.root=root_node
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
            if symbol in ['+', '-', '*', '/', '^','=',SpecialTokens.BRG_TOKEN,SpecialTokens.OPT_TOKEN]:
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

class GoldTree(AbstractTree):
    def __init__(self,root_node=None,gold_ans=None):
        super().__init__()
        self.root=root_node
        self.gold_ans=gold_ans
    def equ2tree(self, equ_list, out_idx2symbol, op_list,num_list,ans):
        stack = []
        for idx in equ_list:
            if idx == out_idx2symbol.index(SpecialTokens.PAD_TOKEN):
                break
            if idx == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                break
            symbol=out_idx2symbol[idx]
            if symbol in op_list:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                if symbol in NumMask.number:
                    i=NumMask.number.index(symbol)
                    value=num_list[i]
                    node = Node(value,isleaf=True)
                elif symbol == SpecialTokens.UNK_TOKEN:
                    node = Node('-inf',isleaf=True)
                else:
                    node = Node(symbol, isleaf=True)
                stack.append(node)
        self.root = stack.pop()
        self.gold_ans = ans
    def is_float(self, num_str,num_list):
        if num_str in num_list:
            return True
        else:
            return False
    def is_equal(self, v1, v2):
        if v1==v2:
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
        if not self.result  and (left+right+mid) == 2:
            if mid:
                self.result = parent
            else:
                self.result = root
        return left or mid or right
    def is_in_rel_quants(self, value,rel_quants):
        if value in rel_quants:
            return True
        else:
            return False 
    def query(self, va, vb):
        if self.root == None:
            return None
        self.result = None
        self.lca(self.root, va, vb, None )
        if self.result:
            return self.result.node_value
        else:
            return self.result