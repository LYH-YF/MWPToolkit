import torch
from torch.nn import functional as F
import numpy as np
from mwptoolkit.utils.utils import get_weakly_supervised
import queue as Q
import numpy as np
import math
import time
import signal
import re

# sym2priority = {'+': 0, '-': 0, '*': 1, '/': 1}
# sym2priority.update({str(x):2 for x in digit_list})

# NAN_THRESHOLD = 10e7
# thres_nan = lambda x: x if abs(eval(x)) < NAN_THRESHOLD else float('nan')
# plus = lambda x,y: thres_nan(eval(x) + eval(y))
# minus = lambda x,y: thres_nan(eval(x) - eval(y))
# times = lambda x,y: thres_nan(eval(x) * eval(y))
# divide = lambda x,y: thres_nan(eval(x) / eval(y) if eval(y) != 0 else float('nan'))
# exp = lambda x,y: thres_nan(eval(x) ** eval(y) if abs(eval(x)) < 10000 and eval(y) <1000 else float('nan'))
# root = lambda x,y: thres_nan(exp(eval(x), divide(1, eval(y))))
# log = lambda x,base: thres_nan(math.log(eval(x), base) if base != 0 and base != 1 and eval(x) > 0 else float('nan'))
# symbol2semantic= {'+': plus, '-': minus, '*': times, '/': divide, '^': exp}
# #symbol2semantic.update({x: eval(x) if x.isdigit()})
# inverse_op_left = {'+': minus, '-': plus, '*': divide, '/': times, '^': root}
# inverse_op_right = {
#     '+': minus,
#     '-': lambda target, left: minus(left, target),
#     '*': divide,
#     '/': lambda target, left: divide(left, target),
#     '^': log}

NAN_THRESHOLD = 10e7
thres_nan = lambda x: x if abs(x) < NAN_THRESHOLD else float('nan')
plus = lambda x, y: thres_nan(x + y)
minus = lambda x, y: thres_nan(x - y)
times = lambda x, y: thres_nan(x * y)
divide = lambda x, y: thres_nan(x / y if y != 0 and y != 1 else float('nan'))
exp = lambda x, y: thres_nan(x ** y if abs(x) < 1000 and abs(y) < 10 and x != 1 and y != 1 else float('nan'))
root = lambda x, y: thres_nan(exp(x, divide(1, y)))
log = lambda x, base: thres_nan(math.log(x, base) if base > 0 and base != 1 and x > 0 else float('nan'))
# NAN_THRESHOLD = 10e7
# thres_nan = lambda x: x if abs(x) < NAN_THRESHOLD else 1e5
# plus = lambda x,y: thres_nan(x + y)
# minus = lambda x,y: thres_nan(x - y)
# times = lambda x,y: thres_nan(x * y)
# divide = lambda x,y: thres_nan(x / y if y != 0 else 1e5)
# exp = lambda x,y: thres_nan(x ** y if abs(x) < 1000 and y < 10 and x != 1 and y != 1 else 1e5)
# root = lambda x,y: thres_nan(exp(x, divide(1, y)))
# log = lambda x,base: thres_nan(math.log(x, base) if base > 0 and base != 1 and x > 0 else 1e5)
symbol2semantic = {'+': plus, '-': minus, '*': times, '/': divide, '^': exp, '**': exp}
inverse_op_left = {'+': minus, '-': plus, '*': divide, '/': times, '^': root, '**': root}
inverse_op_right = {
    '+': minus,
    '-': lambda target, left: minus(left, target),
    '*': divide,
    '/': lambda target, left: divide(left, target),
    '^': log,
    '**': log}


class LeafNode:
    def __init__(self, symbol, all_prob, sym_list, num_start):
        self.symbol = symbol
        self.all_prob = all_prob - np.log(np.sum(np.exp(all_prob)))
        self.sym_list = sym_list
        self.num_start = num_start
        self.initialize()

    def initialize(self):

        self.symbol_id = self.sym_list.index(self.symbol)
        self.prob = self.all_prob[self.symbol_id]
        self.max_prob = self.all_prob.max()
        self.parent = None
        if self.symbol in symbol2semantic:
            self._res = symbol2semantic[self.symbol]
        else:
            self._res = self.symbol

    def res(self):
        return [self._res, self.prob, self.max_prob]

    def entropy(self):
        return -1 * np.sum(np.exp(self.all_prob) * self.all_prob)

    def sample(self):
        # self.all_prob[self.symbol_id] = np.log(1e-30)
        # self.all_prob = self.all_prob - np.log(np.sum(np.exp(self.all_prob)))

        all_prob = np.exp(self.all_prob)
        all_prob_new = all_prob
        if self.symbol in symbol2semantic:
            all_prob_new[self.num_start:] = 0
        else:
            all_prob_new[:self.num_start] = 0
        all_prob_new[self.sym_list.index(self.symbol)] = 1e-6
        all_prob_new /= all_prob_new.sum()
        new_symbol = np.random.choice(self.sym_list, p=all_prob_new)

        if isinstance(new_symbol, str) and any(char.isdigit() for char in new_symbol):
            new_symbol = float(new_symbol)

        self.prev_symbol = self.symbol
        self.symbol = new_symbol

        self.initialize()
        return self.symbol

    def resume(self):
        self.symbol = self.prev_symbol
        self.initialize()


class Node:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        self.parent = None
        self._res = None  # (res, prob, max_prob)
        self.prob = None
        self.max_prob = None

    def res(self):
        if self._res != None:
            return self._res
        left_res = self.left.res()
        right_res = self.right.res()

        op_res = self.op.res()
        prob = left_res[1] + right_res[1] + op_res[1]
        max_prob = left_res[2] + right_res[2] + op_res[2]
        try:
            res = op_res[0](left_res[0], right_res[0])
        except:
            res = float('nan')
        self._res = [res, prob, max_prob]
        self.prob = prob
        self.max_prob = max_prob
        return self._res

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class ExprTree:
    def __init__(self, sym_list, num_start):
        self.tokens = None
        self.root = None
        self.sym_list = sym_list
        self.num_start = num_start

    def handeler(self, signo, frame):
        print("runtime error")
        raise RuntimeError

    def parse(self, tokens=None):
        if tokens is not None:
            tokens = [LeafNode(*tok, self.sym_list, self.num_start) for tok in tokens]
            self.tokens = tokens
        else:
            tokens = self.tokens

        values = []
        operators = []

        # for token in tokens:
        #     if token.symbol in ["+", "-", "*", "/", "^", "**"]:
        #         operators.append(token)
        #     else:
        #         values.append(token)
        #         while len(values) == 2:
        #             op = operators.pop()
        #             right = values.pop()
        #             left = values.pop()
        #             new_node = Node(left, right, op)
        #             op.parent = new_node
        #             right.parent = new_node
        #             left.parent = new_node
        #             values.append(new_node)

        for token in reversed(tokens):
            if token.symbol not in ["+", "-", "*", "/", "^", "**"]:
                values.append(token)
            else:
                op = token
                left = values.pop()
                right = values.pop()
                new_node = Node(left, right, op)
                op.parent = new_node
                right.parent = new_node
                left.parent = new_node
                values.append(new_node)
        # for token in tokens:
        #     if token.symbol in digit_list:
        #         values.append(token)
        #     else:
        #         while len(operators) > 0 and operators[-1].priority >= token.priority:
        #             op = operators.pop()
        #             right = values.pop()
        #             left = values.pop()
        #             new_node = Node(left, right, op)
        #             op.parent = new_node
        #             right.parent = new_node
        #             left.parent = new_node
        #             values.append(new_node)
        #         operators.append(token)

        # while len(operators) > 0:
        #     op = operators.pop()
        #     right = values.pop()
        #     left = values.pop()
        #     new_node = Node(left, right, op)
        #     op.parent = new_node
        #     right.parent = new_node
        #     left.parent = new_node
        #     values.append(new_node)

        self.root = values.pop()
        self.root.res()
        return self.root

    def res(self):
        return self.root.res()

    # def find_valid_change(self, node, target):
    #     if isinstance(node, LeafNode):
    #         target = round(target, 3)
    #         if target in list(map(int, digit_list)):
    #             target = str(int(target))
    #             target_id = sym2id(target)
    #             change = PrioritizedItem(node.prob - node.all_prob[target_id], (node, target))
    #         else:
    #             change = None
    #     else:
    #         change = PrioritizedItem(node.prob - node.max_prob, (node, target))
    #     return change

    def find_valid_change(self, node, target, op):
        if isinstance(node, LeafNode):
            find = False
            for sym in self.sym_list:
                if not isinstance(sym, str):
                    if not (op == "**" and sym == 1):
                        if abs(target - sym) < 1e-7:
                            change = PrioritizedItem(node.prob - node.all_prob[self.sym_list.index(sym)],
                                                     (node, target, sym))
                            find = True
            if not find:
                change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change

    # def prefix_to_infix(self, formula):
    #     stack = []
    #     #prev_op = None
    #     #PRIORITY = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 1, "**": 1}
    #     for ch in reversed(formula):
    #         if not ch in ["+", "-", "*", "/", "^", "**"]:
    #             stack.append(ch)
    #         else:
    #             a = stack.pop()
    #             b = stack.pop()
    #             #if prev_op and PRIORITY[prev_op] < PRIORITY[ch]:
    #             exp = '('+a+ch+b+')'
    #             # else:
    #             #     exp = a+ch+b
    #             stack.append(exp)
    #             # prev_op = ch
    #     return stack[-1]

    def compute_prefix_expression(self, pre_fix):
        st = list()
        operators = ["+", "-", "**", "*", "/"]
        pre_fix.reverse()
        try:
            for p in pre_fix:
                if p not in operators:
                    pos = re.search("\d+\(", p)
                    if pos:
                        st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                    elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                    else:
                        st.append(eval(p))
                elif p == "+" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a + b)
                elif p == "*" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a * b)
                elif p == "/" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if b == 0 or b == 1:
                        return None
                    st.append(a / b)
                elif p == "-" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a - b)
                elif p == "**" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if float(b) != 2.0 or float(b) != 3.0:
                        return None
                    st.append(a ** b)
                else:
                    return None
        except:
            return None
        if len(st) == 1:
            return st.pop()
        return None

    def fix_1step(self, gt):
        # queue = Q.PriorityQueue()
        # change = PrioritizedItem(0., (self.root, gt))
        # queue.put(change)
        # find_fix = False
        # while not queue.empty():
        #     change = queue.get()
        #     prob = change.priority
        #     node, target = change.item
        #     if isinstance(node, LeafNode):
        #         # print('find a fix, early stop.')
        #         find_fix = True
        #         break

        #     left = node.left
        #     right = node.right
        #     op = node.op

        #     # change left
        #     sub_target = inverse_op[op.res()[0]](target, right.res()[0])
        #     change = self.find_valid_change(left, sub_target)
        #     if change != None:
        #         queue.put(change)

        #     # change right
        #     if op.symbol in ['+', '*']:
        #         sub_target = inverse_op[op.res()[0]](target, left.res()[0])
        #     else:
        #         sub_target = op.res()[0](left.res()[0], target)
        #     change = self.find_valid_change(right, sub_target)
        #     if change != None:
        #         queue.put(change)

        #     # change op
        #     ori_op = op.symbol
        #     token_id = self.tokens.index(op)
        #     sub_target = None
        #     for new_op in op_list:
        #         if new_op == ori_op:
        #             continue
        #         new_str = [tok.symbol for tok in self.tokens]
        #         new_str[token_id] = new_op
        #         new_res = eval(''.join(new_str))
        #         if equal_res(new_res, gt):
        #             sub_target = new_op
        #             change = PrioritizedItem(op.prob - op.all_prob[sym2id(sub_target)], (op, sub_target))
        #             queue.put(change)

        # if find_fix:
        #     token_id = self.tokens.index(node)
        #     new_str = [tok.symbol for tok in self.tokens]
        #     if not isinstance(target, str):
        #         target = str(int(target))
        #     new_str[token_id] = target
        #     return (new_str, self.root.res()[1] - prob)
        olds = [tok.symbol for tok in self.tokens]

        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)

        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target, *rest = change.item
            if isinstance(node, LeafNode):
                # print('find a fix, early stop.')
                token_idx = self.tokens.index(node)

                if len(change.item) >= 3:  # if target_sym exists
                    target_sym = change.item[2]
                    news = olds.copy()
                    news[token_idx] = target_sym
                    return (news, self.root.res()[1] - prob)
                else:
                    return None

            left = node.left
            right = node.right
            op = node.op

            if right.res()[0] == float('nan') or left.res()[0] == float('nan'):
                return None
            # change left
            try:
                sub_target = inverse_op_left[op.symbol](target, right.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change(left, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                # if DEBUG and len(change.item) >= 3:
                #     changed_token_ids = old_ids.copy()
                #     changed_idx = self.tokens.index(left)
                #     changed_token_ids[changed_idx] = change.item[2]
                #     print(f"    try change: {self.token_id_list_to_str(changed_token_ids)}")

                queue.put(change)

            # change right
            try:
                sub_target = inverse_op_right[op.symbol](target, left.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change(right, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                #     if DEBUG and len(change.item) >= 3:
                #         changed_token_ids = old_ids.copy()
                #         changed_idx = self.tokens.index(right)
                #         changed_token_ids[changed_idx] = change.item[2]
                #         print(f"    try change: {self.token_id_list_to_str(changed_token_ids)}")

                queue.put(change)

            # change op
            ori_op = op.symbol
            token_idx = self.tokens.index(op)
            sub_target = None

            for new_op in symbol2semantic.keys():
                if new_op == ori_op:
                    continue

                new_exp = [tok.symbol for tok in self.tokens]
                new_exp[token_idx] = "**" if new_op == "^" else new_op
                for j in range(len(new_exp)):
                    if not isinstance(new_exp[j], str):
                        new_exp[j] = str(new_exp[j])

                # new_str = self.prefix_to_infix(new_exp)

                # start = time.time()
                # future = time.time() + 3e-5
                # try:
                # signal.signal(signal.SIGALRM, self.handler)
                # signal.alarm(0.1)

                # new_res = eval(''.join(new_str))
                # for idx in range(new_exp):
                #     if not isinstance (new_exp[idx], str):
                #         new_exp

                new_res = self.compute_prefix_expression(new_exp)
                # print (new_res)
                if not new_res:
                    continue
                if abs(new_res - gt) < 1e-5:
                    sub_target = new_op
                    change = PrioritizedItem(op.prob - op.all_prob[self.sym_list.index(sub_target)],
                                             (op, sub_target, sub_target))

                    # if DEBUG and len(change.item) >= 3:
                    #     changed_token_ids = old_ids.copy()
                    #     changed_token_ids[token_idx] = change.item[2]
                    #     print(f"    try change op: {self.token_id_list_to_str(changed_token_ids)}")

                    queue.put(change)
                # except:
                #     pass

        return None

    def fix(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        res_list = []

        for i in range(n_step):
            if i > 0:
                self.parse()
                # results = [tok.symbol for tok in self.tokens]
                # # res = [tok._res for tok in self.tokens]
                # print (results)
                # # print (res)
                # print (self.res())
            fix = self.fix_1step(gt)

            if fix is not None:
                return fix
            else:
                accept = False
                not_accept_times = 0
                while not accept and not_accept_times <= 5:
                    not_accept_times += 1
                    n_sym_change = int(np.abs(np.random.normal(0, 1, 1)))
                    n_sym_change = np.maximum(n_sym_change, 1)
                    n_sym_change = np.minimum(n_sym_change, len(self.tokens))

                    prob_old_string = np.sum([x.prob for x in self.tokens])
                    token_ids = np.random.choice(len(self.tokens), n_sym_change, replace=False)
                    results = [tok.symbol for tok in self.tokens]
                    for tok_id in token_ids:
                        self.tokens[tok_id].sample()
                    prob_new_string = np.sum([x.prob for x in self.tokens])
                    accept_ratio = np.exp(prob_new_string - prob_old_string)
                    if np.random.random() < accept_ratio:
                        results = [tok.symbol for tok in self.tokens]
                        if results not in res_list:
                            res_list.append(results)
                            accept = True
                        else:
                            accept = False
                            for tok_id in token_ids:
                                self.tokens[tok_id].resume()
                    else:
                        for tok_id in token_ids:
                            self.tokens[tok_id].resume()

        return None

    def fix_bak(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        for i in range(n_step):
            if i > 0:
                self.parse()
            fix = self.fix_1step(gt)
            if fix is not None:
                return fix
            else:
                token_id = np.random.choice(entropy_list.shape[0], p=entropy_list)
                new_symbol = self.tokens[token_id].sample()
        return None


def out_expression_list(test, output_lang, num_list):
    res = []
    for i in test:
        # if i == 0:
        #     return res

        idx = output_lang.dataset.out_idx2symbol[i]

        if "NUM_" in idx:
            if int(idx[4:]) >= len(num_list):
                continue
            res.append(num_list[int(idx[4:])])
        elif "UNK" in idx or "PAD" in idx or 'SOS' in idx:
            continue
        elif "EOS" in idx:
            break
        else:
            res.append(idx)

    return res




def prefix_to_infix(formula, length=None):
    if length is not None:
        formula = formula[:length]
    stack = []
    #prev_op = None
    #PRIORITY = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 1, "**": 1}
    for ch in reversed(formula):
        if not ch in ["+", "-", "*", "/", "^", "**"]:
            stack.append(ch)
        else:
            a = stack.pop()
            b = stack.pop()
            #if prev_op and PRIORITY[prev_op] < PRIORITY[ch]:
            exp = '('+a+ch+b+')'
            #else:
            #    exp = a+ch+b
            stack.append(exp)
            prev_op = ch
    return stack[-1]



def fixStrategy(idx, exp, num, generate_exp, target_length, num_ans, all_list, probs,num_start, n_step, fix_input_length,
                fix_target_list, fix_index, fix_target_length, fix_found, buffer_batch_new, buffer_batch_new_exp,
                input_length, Lang):
    #print("generate_exp[:target_length[idx]]", generate_exp[:target_length[idx]])
    #print("num_ans[idx]", num_ans[idx])
    #print("probs", probs.size())
    #print("all_list", all_list)
    #print("num_start", num_start)
    #print("n_step", n_step)
    try:
        fix = find_fix(
            generate_exp[:target_length[idx]],
            num_ans[idx],
            probs,
            all_list,
            num_start,
            n_step)
    except:
        fix = []
    #print("fix", fix)
    #print("num_ans", num_ans[idx])
    #print("*"*100)
    if len(fix):
        fix_found[idx] = True
        fix_exp = out_expression_list(fix, Lang, num)
        #print("fix_exp", fix_exp)
        fix_infix = prefix_to_infix(fix_exp, target_length[idx])
        try:
            y = eval(fix_infix)
            if y == eval(num_ans[idx]):
                #print("fix_infix", fix_infix)
                #print("fix", fix)
                #print("num", num)
                #print("y", y)
                #print("gold_ans",eval(num_ans[idx]))
                fix_target_list.append(fix)
                fix_index.append(idx)
                fix_target_length.append(len(fix))
                fix_input_length.append(input_length[idx])

        except:
            pass
    return fix_input_length, fix_target_list,fix_index, fix_target_length,\
        fix_found,buffer_batch_new, buffer_batch_new_exp

def mafixStrategy(idx, exp, num,generate_exp, target_length, num_ans, all_list, probs,num_start, n_step, fix_input_length,
                fix_target_list, fix_index, fix_target_length, fix_found, buffer_batch_new, buffer_batch_new_exp,
                  input_length, Lang):
    try:
        fix = find_fix(
            generate_exp[:target_length[idx]],
            num_ans[idx],
            probs,
            all_list,
            num_start,
            n_step)
    except:
        fix = []

    if len(fix):
        fix_found[idx] = True
        fix_exp = out_expression_list(fix, Lang, num)
        fix_infix = prefix_to_infix(fix_exp, target_length[idx])
        try:
            y = eval(fix_infix)
            if y == eval(num_ans[idx]):
                if not fix in buffer_batch_new[idx]:
                    buffer_batch_new[idx].append(fix)
                    buffer_batch_new_exp[idx].append(fix_infix)
        except:
            pass
    for buffer_fix in buffer_batch_new[idx]:
        fix_target_list.append(buffer_fix)
        fix_index.append(idx)
        fix_target_length.append(len(buffer_fix))
        fix_input_length.append(input_length[idx])

    return fix_input_length, fix_target_list,fix_index, fix_target_length,\
        fix_found,buffer_batch_new, buffer_batch_new_exp


def reinforceStrategy(idx, exp, num,generate_exp, target_length, num_ans, all_list, probs,num_start, n_step, fix_input_length,
                fix_target_list, fix_index, fix_target_length, fix_found, buffer_batch_new, buffer_batch_new_exp,
                      input_length, Lang):
    try:
        generate_infix = prefix_to_infix(generate_exp, target_length[idx])

        if eval(generate_infix) == eval(num_ans[idx]):
            fix_target_list.append(exp.item()[:target_length[idx]])
            fix_index.append(idx)
            fix_target_length.append(len(exp))
            fix_input_length.append(input_length[idx])

    except:
        pass
    return fix_input_length, fix_target_list,fix_index, fix_target_length,\
        fix_found,buffer_batch_new, buffer_batch_new_exp


def mapoStrategy(idx,exp, num,generate_exp, target_length, num_ans, all_list, probs,num_start, n_step, fix_input_length,
                fix_target_list, fix_index, fix_target_length, fix_found, buffer_batch_new, buffer_batch_new_exp,
                 input_length, Lang):
    try:
        generate_infix = prefix_to_infix(generate_exp, target_length[idx])

        if eval(generate_infix) == eval(num_ans[idx]):

            if not exp in buffer_batch_new[idx]:
                buffer_batch_new[idx].append(exp.item()[:target_length[idx]])
                buffer_batch_new_exp[idx].append(generate_infix)
    except:
        pass


    for buffer_fix in buffer_batch_new[idx]:
        fix_target_list.append(buffer_fix)
        fix_index.append(idx)
        fix_target_length.append(len(buffer_fix))
        fix_input_length.append(input_length[idx])
    return fix_input_length, fix_target_list,fix_index, fix_target_length,\
        fix_found,buffer_batch_new, buffer_batch_new_exp



def find_fix(pred, gt, all_prob, sym_list, num_start, n_step):
    """
    preds: batch_size * expr len                 int - predicted ids
    res: batch_size                              float - labeled correct result
    probs: batch_size * expr len * classes       float - predicted all probabilities
    num_list: batch_size * list
    """
    try:
        gt = eval(gt)

        for i in range(len(pred)):
            if any(char.isdigit() for char in pred[i]):
                pred[i] = eval(pred[i].replace("%", "/100"))
            if pred[i] == "^":
                pred[i] = "**"

        for i in range(len(sym_list)):
            if any(char.isdigit() for char in sym_list[i]):
                sym_list[i] = eval(sym_list[i].replace("%", "/100"))
            if sym_list[i] == "^":
                sym_list[i] = "**"
    except:
        return []

    tokens = list(zip(pred, all_prob))
    etree = ExprTree(sym_list, num_start)
    etree.parse(tokens)
    fix = []
    try:
        if abs(etree.res()[0] - gt) <= 1e-5:
            fix = [sym_list.index(i) for i in pred]
    except TypeError:
        output = etree.fix(gt, n_step=n_step)
        if output:
            fix = [sym_list.index(i) for i in output[0]]
        # print("No fix needed")
    else:
        output = etree.fix(gt, n_step=n_step)
        if output:
            fix = [sym_list.index(i) for i in output[0]]

            #     print(f"  Fix found: {''.join(old_str)} "
            #             f"=> {''.join(new_str)} = {gt}")
            #     print(f"  {output}")
            # print ("fix found")
            # print (gt)
            # print (pred)

    return fix

