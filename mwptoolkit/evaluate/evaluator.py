import copy
import re
from mwptoolkit.utils.enum_type import PAD_TOKEN,EOS_TOKEN,SOS_TOKEN,UNK_TOKEN,OPERATORS
class SeqEvaluater(object):
    def __init__(self,symbol2idx,idx2symbol,config):
        super().__init__()
        self.share_vocab=config["share_vocab"]
        self.symbol2idx=symbol2idx
        self.idx2symbol=idx2symbol
        self.eos_idx=symbol2idx[EOS_TOKEN]
        self.pad_idx=symbol2idx[PAD_TOKEN]
        if self.share_vocab:
            self.sos_idx=None
        else:
            self.sos_idx=symbol2idx[SOS_TOKEN]
    def result(self,test_res,test_tar,num_list,num_stack=None):
        res_exp=self.out_expression(test_res)
        tar_exp=self.out_expression(test_tar)
        if res_exp==tar_exp:
            return True,True,res_exp,tar_exp
        res_ans=self.compute_expression(res_exp,num_list)
        tar_ans=self.compute_expression(tar_exp,num_list)
        if res_ans !=None:
            if abs(res_ans-tar_ans)<1e-4:
                return True,False,res_exp,tar_exp
            else:
                return False,False,res_exp,tar_exp
        else:
            return False,False,res_exp,tar_exp
    def out_expression(self,test):
        expression=[]
        for idx in test:
            if idx in [self.pad_idx,self.eos_idx,self.sos_idx]:
                break
            symbol=self.idx2symbol[idx]
            expression.append(symbol)
        return expression
    def compute_expression(self,expression,num_list):
        alphabet="abcdefghijklmnopqrstuvwxyz"
        list_len=len(num_list)
        equation=[]
        for symbol in expression:
            if "NUM" in symbol:
                idx=symbol[4]
                num_idx=alphabet.index(idx)
                if num_idx>=list_len:
                    return None
                else:
                    num=num_list[num_idx]
                    if "%" in num:
                        num="("+num[:-1]+"/100"+")"
                        equation.append(num)
                    else:
                        equation.append(num_list[num_idx])
            else:
                if symbol=="^":
                    equation.append("**")
                elif symbol=="[":
                    equation.append("(")
                elif symbol=="]":
                    equation.append(")")
                else:
                    equation.append(symbol)
        equation=''.join(equation)
        try:
            ans=eval(equation)
            return ans
        except:
            return None

class Evaluater(object):
    def __init__(self,symbol2idx,idx2symbol,config):
        super().__init__()
        self.symbol2idx=symbol2idx
        self.idx2symbol=idx2symbol
    def result(self,test_res,test_tar,num_list,num_stack):
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
        alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len=len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i < max_index - 1:
                symbol = self.idx2symbol[i]
                if "NUM" in symbol:
                    idx=symbol[4]
                    num_idx=alphabet.index(idx)
                    if num_idx >= num_len:
                        return None
                    res.append(num_list[num_idx])
                else:
                    res.append(symbol)
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
