import copy
import re
import sympy as sym
from mwptoolkit.utils.enum_type import SpecialTokens,OPERATORS,NumMask,MaskSymbol
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix

class AbstractEvaluater(object):
    def __init__(self,symbol2idx,idx2symbol,config):
        super().__init__()
        self.share_vocab=config["share_vocab"]
        self.mask_symbol=config["mask_symbol"]
        self.symbol2idx=symbol2idx
        self.idx2symbol=idx2symbol
        self.task_type=config["task_type"]

        if self.mask_symbol==MaskSymbol.NUM:
            self.mask_list=NumMask.number
        elif self.mask_symbol==MaskSymbol.alphabet:
            self.mask_list=NumMask.alphabet
        elif self.mask_symbol==MaskSymbol.number:
            self.mask_list=NumMask.number
        else:
            raise NotImplementedError
        try:
            self.eos_idx=symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.eos_idx=None
        try:
            self.pad_idx=symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.pad_idx=None
        try:
            self.sos_idx=symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.sos_idx=None
        try:
            self.unk_idx=symbol2idx[SpecialTokens.UNK_TOKEN]
        except:
            self.unk_idx=None
    def result(self):
        raise NotImplementedError

class SeqEvaluater(AbstractEvaluater):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)
    
    def result(self,test_res,test_tar,num_list,num_stack):
        r'''evaluate single equation'''
        res_exp=self.out_expression_list(test_res,num_list,copy.deepcopy(num_stack))
        tar_exp=self.out_expression_list(test_tar,num_list,copy.deepcopy(num_stack))
        if res_exp==None:
            return False,False,res_exp,tar_exp
        if res_exp==tar_exp:
            return True,True,res_exp,tar_exp
        try:
            if abs(self.compute_expression_by_postfix(res_exp) - self.compute_expression_by_postfix(tar_exp)) < 1e-4:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp
    
    def result_multi(self,test_res,test_tar,num_list,num_stack):
        r'''evaluate multiple euqations'''
        res_exp=self.out_expression_list(test_res,num_list,copy.deepcopy(num_stack))
        tar_exp=self.out_expression_list(test_tar,num_list,copy.deepcopy(num_stack))
        if res_exp==None:
            return False, False, tar_exp, tar_exp
        if res_exp==tar_exp:
            return True,True,res_exp,tar_exp
        try:
            test_solves,test_unk=self.compute_expression_by_postfix_multi(res_exp)
            tar_solves,tar_unk=self.compute_expression_by_pistfix_multi(res_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, tar_exp, tar_exp
            flag=False
            if len(tar_unk) == 1:
                if len(tar_solves)==1:
                    test_ans=test_solves[list(test_unk.values())[0]]
                    tar_ans=tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans-tar_ans) < 1e-4:
                        flag=True
                else:
                    flag=True
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        if abs(test_ans[0]-tar_ans[0]) > 1e-4:
                            flag=False
                            break
                        
            else:
                if len(tar_solves)==len(tar_unk):
                    flag=True
                    for test_x,tar_x in zip(test_unk.values(),tar_unk.values()):
                        test_ans=test_solves[test_x]
                        tar_ans=tar_solves[tar_x]
                        if abs(test_ans-tar_ans) > 1e-4:
                            flag=False
                            break
                else:
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        try:
                            te_ans=float(test_ans[0])

                        except:
                            te_ans=float(test_ans[1])
                        try:
                            ta_ans=float(tar_ans[0])
                        except:
                            ta_ans=float(tar_ans[1])
                        if abs(te_ans-ta_ans) > 1e-4:
                            flag=False
                            break
            if flag == True:
                return True, False,tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp
        return False, False, tar_exp, tar_exp
    
    def out_expression_list(self,test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len=len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx,self.eos_idx,self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx=self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol==SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        return res
    
    def compute_postfix_expression(self,post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
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
                if a == 0:
                    return None
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(b) != 2.0 and float(b) != 3.0:
                    return None
                st.append(a ** b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None
    
    def compute_postfix_expression_multi(self,post_exp):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols={}
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x=sym.symbols(p)
                        st.append(x)
                        unk_symbols[p]=x
                else:
                    st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b + a)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b * a)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if a == 0:
                    return None,unk_symbols
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None,unk_symbols
                st.append(b ** a)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(b,a)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(b+a)
            else:
                return None,unk_symbols
        if len(st) == 1:
            equations=st.pop()
            unk_list=list(self.unk_symbols.values())
            result=sym.solve(equations,unk_list)
            return result,self.unk_symbols
        return None,unk_symbols
    
    def compute_expression_by_postfix(self,expression):
        try:
            post_exp=from_infix_to_postfix(expression)
        except:
            return None
        return self.compute_postfix_expression(post_exp)
    
    def compute_expression_by_postfix_multi(self,expression):
        try:
            post_exp=from_infix_to_postfix(expression)
        except:
            return None,None
        return self.compute_postfix_expression_multi(post_exp)
    
    def eval_source(self,test_res,test_tar,num_list,num_stack):
        new_test_res=[]
        for symbol in test_res:
            try:
                number=eval(symbol)
                flag=True
            except:
                flag=False
            if symbol in OPERATORS:
                new_test_res.append(symbol)
            elif symbol in NumMask.alphabet:
                new_test_res.append(symbol)
            elif flag == True:
                new_test_res.append(symbol)
            elif symbol in ['(',')','[',']']:
                new_test_res.append(symbol)
            else:
                break
        res_ans=self.compute_expression(new_test_res,num_list)
        tar_ans=self.compute_expression(test_tar,num_list)
        if res_ans !=None:
            try:
                if abs(res_ans-tar_ans)<1e-4:
                    return True,False,new_test_res,test_tar
                else:
                    return False,False,new_test_res,test_tar
            except:
                return False,False,new_test_res,test_tar
        else:
            return False,False,new_test_res,test_tar
    
class PreEvaluater(AbstractEvaluater):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)
    
    def result(self,test_res,test_tar,num_list,num_stack):
        test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
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
    
    def result_multi(self,test_res,test_tar,num_list,num_stack):
        test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        #test = tar
        #print(test, tar)
        if test is None:
            return False, False, None, tar
        if test == tar:
            return True, True, None, tar
        try:
            test_solves,test_unk=self.compute_prefix_expression_multi(test)
            tar_solves,tar_unk=self.compute_prefix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar 
            flag=False
            if len(tar_unk) == 1:
                if len(tar_solves)==1:
                    test_ans=test_solves[list(test_unk.values())[0]]
                    tar_ans=tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans-tar_ans) < 1e-4:
                        flag=True
                else:
                    flag=True
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        if abs(test_ans[0]-tar_ans[0]) > 1e-4:
                            flag=False
                            break
                        
            else:
                if len(tar_solves)==len(tar_unk):
                    flag=True
                    for tar_x in list(tar_unk.values()):
                        test_ans=test_solves[tar_x]
                        tar_ans=tar_solves[tar_x]
                        if abs(test_ans-tar_ans) > 1e-4:
                            flag=False
                            break
                else:
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        try:
                            te_ans=float(test_ans[0])
                        except:
                            te_ans=float(test_ans[1])
                        try:
                            ta_ans=float(tar_ans[0])
                        except:
                            ta_ans=float(tar_ans[1])
                        if abs(te_ans-ta_ans) > 1e-4:
                            flag=False
                            break
            if flag == True:
                return True, False,test, tar 
            else:
                return False, False, test, tar 
        except:
            return False, False, test, tar 
        return False, False, test, tar
    
    def out_expression_list(self,test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len=len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx,self.eos_idx,self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx=self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol==SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
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
                if float(b) != 2.0 and float(b) != 3.0:
                    return None
                st.append(a ** b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None
    
    def compute_prefix_expression_multi(self,prefix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols={}
        pre_fix = copy.deepcopy(pre_fix)
        pre_fix.reverse()
        for p in pre_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x=sym.symbols(p)
                        st.append(x)
                        unk_symbols[p]=x
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
                if float(b) != 2.0 and float(b) != 3.0:
                    return None
                st.append(a ** b)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(a,b)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(a+b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None
    
    def eval_source(self,test_res,test_tar,num_list,num_stack=None):
        raise NotImplementedError

class PostEvaluater(AbstractEvaluater):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)
    
    def result(self,test_res,test_tar,num_list,num_stack):
        test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is None:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            if abs(self.compute_postfix_expression(test) - self.compute_postfix_expression(tar)) < 1e-4:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
    
    def result_multi(self,test_res,test_tar,num_list,num_stack):
        test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        #test = tar
        #print(test, tar)
        if test is None:
            return False, False, None, tar
        if test == tar:
            return True, True, None, tar
        try:
            test_solves,test_unk=self.compute_postfix_expression_multi(test)
            tar_solves,tar_unk=self.compute_postfix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar 
            flag=False
            if len(tar_unk) == 1:
                if len(tar_solves)==1:
                    test_ans=test_solves[list(test_unk.values())[0]]
                    tar_ans=tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans-tar_ans) < 1e-4:
                        flag=True
                else:
                    flag=True
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        if abs(test_ans[0]-tar_ans[0]) > 1e-4:
                            flag=False
                            break
                        
            else:
                if len(tar_solves)==len(tar_unk):
                    flag=True
                    for tar_x in list(tar_unk.values()):
                        test_ans=test_solves[tar_x]
                        tar_ans=tar_solves[tar_x]
                        if abs(test_ans-tar_ans) > 1e-4:
                            flag=False
                            break
                else:
                    for test_ans,tar_ans in zip(test_solves,tar_solves):
                        try:
                            te_ans=float(test_ans[0])

                        except:
                            te_ans=float(test_ans[1])
                        try:
                            ta_ans=float(tar_ans[0])
                        except:
                            ta_ans=float(tar_ans[1])
                        if abs(te_ans-ta_ans) > 1e-4:
                            flag=False
                            break
            if flag == True:
                return True, False,test, tar 
            else:
                return False, False, test, tar 
        except:
            return False, False, test, tar 
        return False, False, test, tar
    
    def out_expression_list(self,test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len=len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx,self.eos_idx,self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx=self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol==SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        return res
    
    def compute_postfix_expression(self,post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
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
                if a == 0:
                    return None
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(b) != 2.0 and float(b) != 3.0:
                    return None
                st.append(a ** b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None
        
    def compute_postfix_expression_multi(self,post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols={}
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x=sym.symbols(p)
                        st.append(x)
                        unk_symbols[p]=x
                else:
                    st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b + a)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b * a)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if a == 0:
                    return None,unk_symbols
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None,unk_symbols
                st.append(b ** a)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(b,a)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(b+a)
            else:
                return None,unk_symbols
        if len(st) == 1:
            equations=st.pop()
            unk_list=list(self.unk_symbols.values())
            result=sym.solve(equations,unk_list)
            return result,self.unk_symbols
        return None,unk_symbols
    
    def eval_source(self):
        raise NotImplementedError


# class SeqEvaluater(object):
#     def __init__(self,symbol2idx,idx2symbol,config):
#         super().__init__()
#         self.share_vocab=config["share_vocab"]
#         self.mask_symbol=config["mask_symbol"]
#         self.symbol2idx=symbol2idx
#         self.idx2symbol=idx2symbol
#         self.eos_idx=symbol2idx[SpecialTokens.EOS_TOKEN]
#         self.pad_idx=symbol2idx[SpecialTokens.PAD_TOKEN]
#         if self.share_vocab:
#             self.sos_idx=None
#         else:
#             self.sos_idx=symbol2idx[SpecialTokens.SOS_TOKEN]
    
#     def result(self,test_res,test_tar,num_list,num_stack=None):
#         res_exp=self.out_expression(test_res)
#         tar_exp=self.out_expression(test_tar)
#         if res_exp==tar_exp:
#             return True,True,res_exp,tar_exp
#         res_ans=self.compute_expression(res_exp,num_list)
#         tar_ans=self.compute_expression(tar_exp,num_list)
#         if res_ans !=None:
#             try:
#                 if abs(res_ans-tar_ans)<1e-4:
#                     return True,False,res_exp,tar_exp
#                 else:
#                     return False,False,res_exp,tar_exp
#             except:
#                 return False,False,res_exp,tar_exp
#         else:
#             return False,False,res_exp,tar_exp
#     def eval_source(self,test_res,test_tar,num_list,num_stack=None):
#         new_test_res=[]
#         for symbol in test_res:
#             try:
#                 number=eval(symbol)
#                 flag=True
#             except:
#                 flag=False
#             if symbol in OPERATORS:
#                 new_test_res.append(symbol)
#             elif symbol in NumMask.alphabet:
#                 new_test_res.append(symbol)
#             elif flag == True:
#                 new_test_res.append(symbol)
#             elif symbol in ['(',')','[',']']:
#                 new_test_res.append(symbol)
#             else:
#                 break
#         res_ans=self.compute_expression(new_test_res,num_list)
#         tar_ans=self.compute_expression(test_tar,num_list)
#         if res_ans !=None:
#             try:
#                 if abs(res_ans-tar_ans)<1e-4:
#                     return True,False,new_test_res,test_tar
#                 else:
#                     return False,False,new_test_res,test_tar
#             except:
#                 return False,False,new_test_res,test_tar
#         else:
#             return False,False,new_test_res,test_tar
#     def out_expression(self,test):
#         expression=[]
#         for idx in test:
#             if idx in [self.pad_idx,self.eos_idx,self.sos_idx]:
#                 break
#             symbol=self.idx2symbol[idx]
#             expression.append(symbol)
#         return expression
    
#     def compute_expression(self,expression,num_list):
#         alphabet="abcdefghijklmnopqrstuvwxyz"
#         list_len=len(num_list)
#         equation=[]
#         for symbol in expression:
#             if "NUM" in symbol:
#                 idx=symbol[4]
#                 num_idx=alphabet.index(idx)
#                 if num_idx>=list_len:
#                     return None
#                 else:
#                     num=num_list[num_idx]
#                     if "%" in num:
#                         num="("+num[:-1]+"/100"+")"
#                         equation.append(num)
#                     else:
#                         equation.append(num_list[num_idx])
#             else:
#                 if symbol=="^":
#                     equation.append("**")
#                 elif symbol=="[":
#                     equation.append("(")
#                 elif symbol=="]":
#                     equation.append(")")
#                 else:
#                     equation.append(symbol)
#         equation=''.join(equation)
#         try:
#             ans=eval(equation)
#             return ans
#         except:
#             return None

# class Evaluater(object):
#     def __init__(self,symbol2idx,idx2symbol,config):
#         super().__init__()
#         self.symbol2idx=symbol2idx
#         self.idx2symbol=idx2symbol
#         try:
#             self.eos_idx=symbol2idx[EOS_TOKEN]
#         except:
#             self.eos_idx=None
#         try:
#             self.pad_idx=symbol2idx[PAD_TOKEN]
#         except:
#             self.pad_idx=None
#         try:
#             self.sos_idx=symbol2idx[SOS_TOKEN]
#         except:
#             self.sos_idx=None
    
#     def result(self,test_res,test_tar,num_list,num_stack):
#         test = self.out_expression_list(test_res, num_list)
#         tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         # print(test, tar)
#         if test is None:
#             return False, False, test, tar
#         if test == tar:
#             return True, True, test, tar
#         try:
#             if abs(self.compute_prefix_expression(test) - self.compute_prefix_expression(tar)) < 1e-4:
#                 return True, False, test, tar
#             else:
#                 return False, False, test, tar
#         except:
#             return False, False, test, tar
    
#     def out_expression_list(self,test, num_list, num_stack=None):
#         alphabet="abcdefghijklmnopqrstuvwxyz"
#         num_len=len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if i in [self.pad_idx,self.eos_idx,self.sos_idx]:
#                 break
#             symbol = self.idx2symbol[i]
#             if "NUM" in symbol:
#                 idx=symbol[4]
#                 num_idx=alphabet.index(idx)
#                 if num_idx >= num_len:
#                     return None
#                 res.append(num_list[num_idx])
#             elif symbol==UNK_TOKEN:
#                 pos_list = num_stack.pop()
#                 c = num_list[pos_list[0]]
#                 res.append(c)
#             else:
#                 res.append(symbol)
#         return res
    
#     def compute_prefix_expression(self,pre_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         pre_fix = copy.deepcopy(pre_fix)
#         pre_fix.reverse()
#         for p in pre_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a + b)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a * b)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a * b)
#             elif p == "/" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if b == 0:
#                     return None
#                 st.append(a / b)
#             elif p == "-" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a - b)
#             elif p == "^" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if float(b) != 2.0 and float(b) != 3.0:
#                     return None
#                 st.append(a ** b)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None
#     def eval_source(self,test_res,test_tar,num_list,num_stack=None):
#         raise NotImplementedError
#         new_test_res=[]
#         for symbol in test_res:
#             try:
#                 number=eval(symbol)
#                 flag=True
#             except:
#                 flag=False
#             if symbol in OPERATORS:
#                 new_test_res.append(symbol)
#             elif symbol in NumMask.alphabet:
#                 new_test_res.append(symbol)
#             elif flag == True:
#                 new_test_res.append(symbol)
#             elif symbol in ['(',')','[',']']:
#                 new_test_res.append(symbol)
#             else:
#                 break
#         if new_test_res == test_tar:
#             return True, True, new_test_res, test_tar
#         try:
#             if abs(self.compute_prefix_expression(new_test_res) - self.compute_prefix_expression(test_tar)) < 1e-4:
#                 return True, False, new_test_res,test_tar
#             else:
#                 return False, False, new_test_res, test_tar
#         except:
#             return False, False, new_test_res, test_tar
# class PostEvaluater(object):
#     def __init__(self,symbol2idx,idx2symbol,config):
#         super().__init__()
#         self.symbol2idx=symbol2idx
#         self.idx2symbol=idx2symbol
#         try:
#             self.eos_idx=symbol2idx[EOS_TOKEN]
#         except:
#             self.eos_idx=None
#         try:
#             self.pad_idx=symbol2idx[PAD_TOKEN]
#         except:
#             self.pad_idx=None
#         try:
#             self.sos_idx=symbol2idx[SOS_TOKEN]
#         except:
#             self.sos_idx=None
    
#     def result(self,test_res,test_tar,num_list,num_stack):
#         test = self.out_expression_list(test_res, num_list)
#         tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         # print(test, tar)
#         if test is None:
#             return False, False, test, tar
#         if test == tar:
#             return True, True, test, tar
#         try:
#             if abs(self.compute_postfix_expression(test) - self.compute_postfix_expression(tar)) < 1e-4:
#                 return True, False, test, tar
#             else:
#                 return False, False, test, tar
#         except:
#             return False, False, test, tar
    
#     def out_expression_list(self,test, num_list, num_stack=None):
#         alphabet="abcdefghijklmnopqrstuvwxyz"
#         num_len=len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if i in [self.pad_idx,self.eos_idx,self.sos_idx]:
#                 break
#             symbol = self.idx2symbol[i]
#             if "NUM" in symbol:
#                 idx=symbol[4]
#                 num_idx=alphabet.index(idx)
#                 if num_idx >= num_len:
#                     return None
#                 res.append(num_list[num_idx])
#             elif symbol==UNK_TOKEN:
#                 pos_list = num_stack.pop()
#                 c = num_list[pos_list[0]]
#                 res.append(c)
#             else:
#                 res.append(symbol)
#         return res
    
#     def compute_postfix_expression(self,post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                         st.append(float(p[:-1]) / 100)
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a + b)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a * b)
#             elif p == "/" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if a == 0:
#                     return None
#                 st.append(b / a)
#             elif p == "-" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b - a)
#             elif p == "^" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if float(b) != 2.0 and float(b) != 3.0:
#                     return None
#                 st.append(a ** b)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None
#     def eval_source(self):
#         raise NotImplementedError