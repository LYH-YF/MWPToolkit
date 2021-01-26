import copy
import re
class Evaluater(object):
    def __init__(self,symbol2idx,idx2symbol):
        super().__init__()
        self.symbol2idx=symbol2idx
        self.idx2symbol=idx2symbol
    
    def prefix_tree_result(self,test_res,test_tar,num_list,num_stack):
        if len(num_stack) == 0 and test_res == test_tar:
            return True, True, test_res, test_tar
        test = self.out_expression_list(test_res, num_list)
        tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is None:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            if abs(self.compute_prefix_expression(test) - self.compute_prefix_expression(tar)) < 1e-4:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
    
    def out_expression_list(self,test, num_list, num_stack=None):
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i < max_index - 1:
                idx = self.idx2symbol[i]
                if idx[0] == "N":
                    if int(idx[1:]) >= len(num_list):
                        return None
                    res.append(num_list[int(idx[1:])])
                else:
                    res.append(idx)
            else:
                pos_list = num_stack.pop()
                c = num_list[pos_list[0]]
                res.append(c)
        return res
    
    def compute_prefix_expression(self,pre_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        pre_fix = copy.deepcopy(pre_fix)
        pre_fix.reverse()
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
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a * b)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if b == 0:
                    return None
                st.append(a / b)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a - b)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                    return None
                st.append(a ** b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None