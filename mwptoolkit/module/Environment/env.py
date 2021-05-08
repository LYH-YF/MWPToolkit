import time
import re

import torch
import numpy as np

from mwptoolkit.utils.utils import str2float


class Env:

    def __init__(self):
        #self.config = config
        self.count = 0
        self.agents = []
        self.curr_agent = None

    # functions for making environment

    def make_env(self,batch_tree,look_up,emb,op_list):
        #self.index_to_feature, self.feature_to_index = self.get_index_to_feature_and_feature_to_index()
        #self.train_set, self.validate_set = self.separate_data_set()
        #equations=batch_data['equation']
        self.count = 0
        self.agents = []
        self.curr_agent = None

        for b_i in range(len(batch_tree)):
            agent = Agent(
                parse_obj={},
                gold_tree=batch_tree[b_i], 
                reject=[], 
                pick=[],
                quantities_emb=emb[b_i],
                look_up=look_up[b_i],
                op_list=op_list)
            self.agents.append(agent)
    
    def reset(self):
        num = self.count
        self.count += 1
        self.curr_agent = self.agents[num]
        self.curr_agent.init_state_info()
        return self.curr_agent.feat_vector
    def validate_reset(self, iteration):
        self.curr_agent = self.agents[iteration]
        self.curr_agent.init_state_info()
        return self.curr_agent.feat_vector

    def separate_data_set(self):
        train_set = []
        validate_set = []
        for ind in self.config.train_list:
            train_set.append(self.agents[ind])
        for ind in self.config.validate_list:
            validate_set.append(self.agents[ind])
        return train_set, validate_set

    # other control functions

    def reset_inner_count(self):
        self.count = 0

    def step(self, action_op):
        next_states, reward, done, flag = self.curr_agent.compound_two_nodes(action_op)
        return next_states, reward, done

    def val_step(self, action_op):
        next_states, done, flag = self.curr_agent.compound_two_nodes_predict(action_op)
        return next_states, done, flag
        

class Node:
    def __init__(self):
        self.is_compound = False
        self.index = []
    
    def init_node(self, i, value):
        self.index.append(i)
        self.value = value

    def is_belong(self, i):
        if i in self.index:
            return True
        return False
 
    def i_and_j_is_belong(self, i, j):
        if i in self.index and j in self.index:
            return True
        return False

    def compute_val(self, v1, v2, op):
        if op == '+':
             return v1 + v2
        elif op == '-':
             return v1 - v2
        elif op == '*':
             return v2 * v1
        elif op == '/':
            try:
                res=v1 / v2
            except:
                res=float('inf')
            return res
        elif op == '^':
            try:
                res=v1 ** v2
            except:
                res=float('inf')
            return res

    def combine_node(self, node1, node2, op):
        self.is_compound = True
        self.op = op
        self.index.extend(node1.index)
        self.index.extend(node2.index)
        value1 = 0
        value2 = 0
        if node1.is_compound == False:
            #value1 = float(node1.value)
            value1 = str2float(node1.value)
        else: 
            value1 = node1.value 
        if node2.is_compound == False:
            value2 = str2float(node2.value)
        else: 
            value2 = node2.value 
        self.value = self.compute_val(value1, value2, op) 

# class State:
#     def __init__(self, quant_tokens):
#         self.nodes = self.get_nodes(quant_tokens) 
#         self.fix_nodes = self.get_nodes(quant_tokens)
#         self.length = len(self.nodes)

#     def str_2_quant(self, word):
#         word = word.lower()
#         l = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
#         return l.index(word)+1

#     def quant_str_2_quant(self, word):
#         try:
#             float(word)
#         except:
#             return str(self.str_2_quant(word)) 
#         else:
#             return word 

#     def get_nodes(self, quant_tokens):
#         nodes = []
#         for i in range(len(quant_tokens)):
#             quant = self.quant_str_2_quant(quant_tokens[i].word_text)
#             node = Node()
#             node.init_node(i, quant)
#             nodes.append(node)
#         return nodes 

#     def get_node_via_index(self, index):
#         for i in range(len(self.nodes)):
#             if self.nodes[i].is_belong(index):
#                 return self.nodes[i]            

#     def is_lca_i_and_j(self, i, j):
#         for node in self.nodes:
#             if node.i_and_j_is_belong(i, j):
#                 return True
#         return False

#     def change(self, i, j, newnode):
#         li = []
#         for node in self.nodes:
#             if node.is_belong(i) or node.is_belong(j):
#                 pass
#             else:
#                 li.append(node)
#         li.append(newnode)
#         self.nodes = li

#     def remove_node(self, i):
#         li = []
#         for node in self.nodes:
#             if node.is_belong(i):
#                 pass
#             else:
#                 li.append(node)
#         self.nodes = li

#     def print_state(self):
#         print("state:", end=' ')
#         s = '['
#         for i in range(len(self.nodes)): 
#             s += '['
#             for ind in self.nodes[i].index:
#                 s += str(self.fix_nodes[ind].value) +', '
#             s += '], '
#         s+= ']' 
#         print(s) 

class State:
    def __init__(self, num_list):
        self.nodes = self.get_nodes(num_list) 
        self.fix_nodes = self.get_nodes(num_list)
        self.length = len(self.nodes)

    def str_2_quant(self, word):
        word = word.lower()
        l = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        return l.index(word)+1

    # def quant_str_2_quant(self, word):
    #     try:
    #         float(word)
    #     except:
    #         return str(self.str_2_quant(word)) 
    #     else:
    #         return word 

    def get_nodes(self, num_list):
        nodes = []
        for i in range(len(num_list)):
            #quant = self.quant_str_2_quant(quant_tokens[i].word_text)
            node = Node()
            node.init_node(i, num_list[i])
            nodes.append(node)
        return nodes 

    def get_node_via_index(self, index):
        for i in range(len(self.nodes)):
            if self.nodes[i].is_belong(index):
                return self.nodes[i]            

    def is_lca_i_and_j(self, i, j):
        for node in self.nodes:
            if node.i_and_j_is_belong(i, j):
                return True
        return False

    def change(self, i, j, newnode):
        li = []
        for node in self.nodes:
            if node.is_belong(i) or node.is_belong(j):
                pass
            else:
                li.append(node)
        li.append(newnode)
        self.nodes = li

    def remove_node(self, i):
        li = []
        for node in self.nodes:
            if node.is_belong(i):
                pass
            else:
                li.append(node)
        self.nodes = li

    def print_state(self):
        print("state:", end=' ')
        s = '['
        for i in range(len(self.nodes)): 
            s += '['
            for ind in self.nodes[i].index:
                s += str(self.fix_nodes[ind].value) +', '
            s += '], '
        s+= ']' 
        print(s) 

class Agent:
    def __init__(self, parse_obj, gold_tree, reject, pick,quantities_emb,look_up,op_list):
        self.parse_obj = parse_obj
        self.gold_tree = gold_tree
        self.reject = reject
        self.pick = pick
        self.quantities_emb=quantities_emb
        self.look_up=look_up
        self.op_list=op_list
        

    def print_agent(self):
        print("index:", self.parse_obj.parse_id)
        print(self.gold_tree.exp_str)

    
    def get_feat_vector(self, index1, index2):
        emb1=self.quantities_emb[index1]
        emb2=self.quantities_emb[index2]
        self.feat_vector=torch.cat([emb1,emb2],dim=-1)

        return self.feat_vector

    def select_tuple(self):
        self.candidate_select = []
        if self.pick != []:
            self.candidate_select.append(self.pick)
        for i in range(self.state.length):
            for j in range(self.state.length):
                if i!= j and i<j and (not (i in self.pick and j in self.pick)):
                    self.candidate_select.append([i,j]) 
        self.reject_select = self.reject 

    def select_combine(self):
        for elem_pair in self.candidate_select:
            if elem_pair[0] in self.reject_select or elem_pair[1] in self.reject_select \
                                                   or self.state.is_lca_i_and_j(elem_pair[0], elem_pair[1]):
                continue
            else:
                return elem_pair 
        return []

    def init_state_info(self):
        self.state = State(self.look_up)
        for index in self.reject:
            self.state.remove_node(index)
        self.select_tuple()
        self.breakout = 0
        #print "candidate:", self.candidate_select
        #print "reject", self.reject_select
        elem_pair = self.select_combine()
        if not elem_pair:
            self.breakout = 1
            #self.feat_dim = len(index_to_feat)
            #self.feat_vector = next_states
            self.feat_vector = torch.randn(self.quantities_emb[0].size(-1)*2)
            return
        self.node_1_index = elem_pair[0] 
        self.node_2_index = elem_pair[1]
        #self.index_to_feat = index_to_feat
        self.get_feat_vector(self.node_1_index, self.node_2_index)

    def compound_two_nodes_predict(self, op):
        op_symbol=self.op_list[op]
        if self.breakout == 1:
            #self.write_single_info(filename, 1, "parse", "_error")
            return None, 1, 0

        self.reward = 0
        node1 = self.state.get_node_via_index(self.node_1_index)
        node2 = self.state.get_node_via_index(self.node_2_index)
        newNode = Node()
        newNode.combine_node(node1, node2, op_symbol)
        self.state.change(self.node_1_index, self.node_2_index, newNode)
        if len(self.state.nodes) == 1:
            if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 1e-4:
                #self.write_single_info(filename, 1, "compute_state_node==1", "_right")
                return None, 1, 1
            else:
                #self.write_single_info(filename, 1, "compute_state_node==1", "_error")
                return None, 1, 0
        elif len(self.state.nodes) == 0:
            #self.write_single_info(filename, 1, "state_node==0", "_error")
            return None, 1, 0
        else:
            elem_pair = self.select_combine()
            self.node_1_index = elem_pair[0]
            self.node_2_index = elem_pair[1]
            next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
            #self.write_single_info(filename, 1, "next", "_step")
            return next_states, 0, 0
            
    def compound_two_nodes(self, op):
        self.reward = 0
        if self.breakout == 1:
            return None, 0, 1, 0

        node1 = self.state.get_node_via_index(self.node_1_index)
        node2 = self.state.get_node_via_index(self.node_2_index)
        fix_node1 = self.state.fix_nodes[self.node_1_index]
        fix_node2 = self.state.fix_nodes[self.node_2_index]
        flag1 = False
        flag2 = False
        if node1.is_compound:
            flag1 =  True
        else:
            if self.gold_tree.is_in_rel_quants(fix_node1.value,self.look_up):
                flag1 = True
            else:
                flag1 = False
        if node2.is_compound:
            flag1 =  True
        else:
            if self.gold_tree.is_in_rel_quants(fix_node2.value,self.look_up):
                flag2 = True
            else:
                flag2 = False
        self.flag1 = flag1
        self.flag2 = flag2
        op_symbol=self.op_list[op]
        if op_symbol == '+':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '+':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '-':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '-':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0 :
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '*':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '*':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0 :
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '/':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '/':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0 :
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '^':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '^':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0 :
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        else:
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '-':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op)
                    if newNode.value < 0 :
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if self.state.nodes[0].node_value == self.gold_tree.gold_ans < 1e-4:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0

        

    def test_gate(self, flag):
        self.test_flag = flag

    def write_info(self, filename, op):
        with open(filename, 'a') as f:
            f.write("index: " + str(self.parse_obj.parse_id) + '\n')
            f.write(self.parse_obj.word_problem_text+'\n')
            f.write("equations: " + str(self.gold_tree.exp_str) + '\n' )
            f.write("node_1: " + str((self.state.fix_nodes[self.node_1_index]).value) + ", node_2: " + str((self.state.fix_nodes[self.node_2_index]).value) + '\n')
            f.write("op: " + (['+','-','in-','11','22','33'][op]) + '\n')
            f.write("gold_ans: " + str(self.gold_tree.gold_ans) + '\n')

    def write_single_info(self, filename, flag, prefix, content):
        if flag:
            with open(filename, 'a') as f:
                f.write(prefix + content + '\n\n')

