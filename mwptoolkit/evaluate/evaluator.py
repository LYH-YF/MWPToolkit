import copy
import re
import threading
import sympy as sym
from mwptoolkit.utils.enum_type import SpecialTokens, OPERATORS, NumMask, MaskSymbol
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix


class Solver(threading.Thread):
    r"""time-limited solving equation machanism based threading.
    """
    def __init__(self, func, equations, unk_symbol):
        super(Solver, self).__init__()
        self.func = func
        self.equations = equations
        self.unk_symbol = unk_symbol

    def run(self):
        try:
            self.result = self.func(self.equations, self.unk_symbol)
        except:
            self.result = None

    def get_result(self):
        try:
            return self.result
        except:
            return None


class AbstractEvaluator(object):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__()
        self.share_vocab = config["share_vocab"]
        self.mask_symbol = config["mask_symbol"]
        self.symbol2idx = symbol2idx
        self.idx2symbol = idx2symbol
        self.task_type = config["task_type"]
        self.single = config["single"]
        self.linear = config["linear"]

        if self.mask_symbol == MaskSymbol.NUM:
            self.mask_list = NumMask.number
        elif self.mask_symbol == MaskSymbol.alphabet:
            self.mask_list = NumMask.alphabet
        elif self.mask_symbol == MaskSymbol.number:
            self.mask_list = NumMask.number
        else:
            raise NotImplementedError
        try:
            self.eos_idx = symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.eos_idx = None
        try:
            self.pad_idx = symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.pad_idx = None
        try:
            self.sos_idx = symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.sos_idx = None
        try:
            self.unk_idx = symbol2idx[SpecialTokens.UNK_TOKEN]
        except:
            self.unk_idx = None

    def result(self):
        raise NotImplementedError

    def result_multi(self):
        raise NotImplementedError


class SeqEvaluator(AbstractEvaluator):
    r"""evaluator for normal equation sequnence.
    """
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)

    def result(self, res_exp, tar_exp):
        r"""evaluate single equation.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(res_exp, tar_exp)
        
        if res_exp == []:
            return False, False, res_exp, tar_exp
        if res_exp == tar_exp:
            return True, True, res_exp, tar_exp
        try:
            if abs(self.compute_expression_by_postfix(res_exp) - self.compute_expression_by_postfix(tar_exp)) < 1e-4:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def result_multi(self, res_exp, tar_exp):
        r"""evaluate multiple euqations.
        """
        #print(test_tar)# tensor([20,  8, 10, 12,  5, 13, 11,  2])
        #res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        #print(tar_exp)# ['x', '=', '(', '4', '*', '3', ')']
        if res_exp == []:
            return False, False, res_exp, tar_exp
        if res_exp == tar_exp:
            return True, True, res_exp, tar_exp
        try:
            test_solves, test_unk = self.compute_expression_by_postfix_multi(res_exp)
            tar_solves, tar_unk = self.compute_expression_by_postfix_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, res_exp, tar_exp
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])

                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def out_expression_list(self, test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len = len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        if res == []:
            return None
        return res

    def compute_postfix_expression(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
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
                    return None
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None
                st.append(b**a)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def compute_postfix_expression_multi(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols = {}
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x = sym.symbols(p)
                        st.append(x)
                        unk_symbols[p] = x
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
                    return None, unk_symbols
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None, unk_symbols
                st.append(b**a)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(b, a)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(b + a)
            else:
                return None, unk_symbols
        if len(st) == 1:
            equations = st.pop()
            unk_list = list(unk_symbols.values())
            t = Solver(sym.solve, equations, unk_list)
            t.setDaemon(True)
            t.start()
            t.join(10)
            result = t.get_result()
            return result, unk_symbols
        return None, unk_symbols

    def compute_expression_by_postfix(self, expression):
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None
        return self.compute_postfix_expression(post_exp)

    def compute_expression_by_postfix_multi(self, expression):
        r"""return solves and unknown number list
        """
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None, None
        return self.compute_postfix_expression_multi(post_exp)

    def eval_source(self, test_res, test_tar, num_list, num_stack):
        num_len = len(num_list)
        new_test_res = []
        for symbol in test_res:
            if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
                break
            elif "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                new_test_res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    new_test_res.append(c)
                except:
                    new_test_res=None
                    break
            else:
                new_test_res.append(symbol)
        new_test_tar = []
        for symbol in test_tar:
            if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
                break
            elif "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                new_test_tar.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    new_test_tar.append(c)
                except:
                    new_test_tar=None
                    break
            else:
                new_test_tar.append(symbol)

        #res_ans = self.compute_expression_by_postfix(new_test_res)
        #tar_ans = self.compute_expression_by_postfix(test_tar, num_list)
        if self.single and self.linear:
            try:
                if abs(self.compute_expression_by_postfix(new_test_res) - self.compute_expression_by_postfix(new_test_tar)) < 1e-4:
                    return True, False, new_test_res, new_test_tar
                else:
                    return False, False, new_test_res, new_test_tar
            except:
                return False, False, new_test_res, new_test_tar
        else:
            try:
                if abs(self.compute_expression_by_postfix_multi(new_test_res) - self.compute_expression_by_postfix_multi(new_test_tar)) < 1e-4:
                    return True, False, new_test_res, new_test_tar
                else:
                    return False, False, new_test_res, new_test_tar
            except:
                return False, False, new_test_res, new_test_tar


class PreEvaluator(AbstractEvaluator):
    r"""evaluator for prefix equation.
    """
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)

    def result(self, test, tar):
        r"""evaluate single equation.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test,tar)
        #test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is []:
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

    def result_multi(self, test,tar):
        r"""evaluate multiple euqations.
        """
        #test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        #test = tar
        #print(test, tar)
        if test is []:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            test_solves, test_unk = self.compute_prefix_expression_multi(test)
            tar_solves, tar_unk = self.compute_prefix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])
                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
        return False, False, test, tar

    def out_expression_list(self, test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len = len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        if res == []:
            return None
        return res

    def compute_prefix_expression(self, pre_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        pre_fix = copy.deepcopy(pre_fix)
        pre_fix.reverse()
        for p in pre_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
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
                st.append(a**b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def compute_prefix_expression_multi(self, pre_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols = {}
        pre_fix = copy.deepcopy(pre_fix)
        pre_fix.reverse()
        for p in pre_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x = sym.symbols(p)
                        st.append(x)
                        unk_symbols[p] = x
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
                st.append(a**b)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(a, b)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(a + b)
            else:
                return None
        if len(st) == 1:
            equations = st.pop()
            unk_list = list(unk_symbols.values())
            t = Solver(sym.solve, equations, unk_list)
            t.setDaemon(True)
            t.start()
            t.join(10)
            result = t.get_result()
            return result, unk_symbols
        return None

    def eval_source(self, test_res, test_tar, num_list, num_stack=None):
        raise NotImplementedError


class PostEvaluator(AbstractEvaluator):
    r"""evaluator for postfix equation.
    """
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)

    def result(self, test,tar):
        r"""evaluate single equation.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test,tar)
        #test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is []:
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

    def result_multi(self, test, tar):
        r"""evaluate multiple euqations.
        """
        #test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        if test is []:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            test_solves, test_unk = self.compute_postfix_expression_multi(test)
            tar_solves, tar_unk = self.compute_postfix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])

                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
        return False, False, test, tar

    def out_expression_list(self, test, num_list, num_stack=None):
        #alphabet="abcdefghijklmnopqrstuvwxyz"
        num_len = len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
                break
            symbol = self.idx2symbol[i]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        if res == []:
            return None
        return res

    def compute_postfix_expression(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                else:
                    st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b + a)
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
                if float(a) != 2.0 and float(a) != 3.0:
                    return None
                st.append(b**a)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def compute_postfix_expression_multi(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols = {}
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x = sym.symbols(p)
                        st.append(x)
                        unk_symbols[p] = x
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
                    return None, unk_symbols
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None, unk_symbols
                st.append(b**a)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(b, a)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(b + a)
            else:
                return None, unk_symbols
        if len(st) == 1:
            equations = st.pop()
            unk_list = list(unk_symbols.values())
            t = Solver(sym.solve, equations, unk_list)
            t.setDaemon(True)
            t.start()
            t.join(10)
            result = t.get_result()
            return result, unk_symbols
        return None, unk_symbols

    def eval_source(self):
        raise NotImplementedError


class MultiWayTreeEvaluator(AbstractEvaluator):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)

    def result(self, res_exp, tar_exp):
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(res_exp, tar_exp)
        #res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        #res_exp = copy.deepcopy(tar_exp)
        if res_exp == []:
            return False, False, res_exp, tar_exp
        if res_exp == tar_exp:
            return True, True, res_exp, tar_exp
        try:
            if abs(self.compute_expression_by_postfix(res_exp) - self.compute_expression_by_postfix(tar_exp)) < 1e-4:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def result_multi(self, res_exp, tar_exp):
        r"""evaluate multiple euqations.
        """
        #res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        if res_exp == []:
            return False, False, res_exp, tar_exp
        if res_exp == tar_exp:
            return True, True, res_exp, tar_exp
        try:
            test_solves, test_unk = self.compute_expression_by_postfix_multi(res_exp)
            tar_solves, tar_unk = self.compute_expression_by_postfix_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, res_exp, tar_exp
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])

                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp
    
    def out_expression_list(self, test, num_list, num_stack=None):
        num_len = len(num_list)
        max_index = len(self.idx2symbol)
        res = []
        for i in test:
            if isinstance(i,list):
                sub_res=self.out_expression_list(i,num_list,num_stack)
                if sub_res==None:
                    return None
                res.append("(")
                res+=sub_res
                res.append(")")
            else:
                if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
                    break
                symbol = self.idx2symbol[i]
                if "NUM" in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        return None
                    res.append(num_list[num_idx])
                elif symbol == SpecialTokens.UNK_TOKEN:
                    try:
                        pos_list = num_stack.pop()
                        c = num_list[pos_list[0]]
                        res.append(c)
                    except:
                        return None
                else:
                    res.append(symbol)
        if res == []:
            return None
        return res
    
    def compute_postfix_expression(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
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
                    return None
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None
                st.append(b**a)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def compute_postfix_expression_multi(self, post_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols = {}
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    if p in unk_symbols:
                        st.append(unk_symbols[p])
                    else:
                        x = sym.symbols(p)
                        st.append(x)
                        unk_symbols[p] = x
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
                    return None, unk_symbols
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(a) != 2.0 and float(a) != 3.0:
                    return None, unk_symbols
                st.append(b**a)
            elif p == "=":
                a = st.pop()
                b = st.pop()
                st.append([sym.Eq(b, a)])
            elif p == "<BRG>":
                a = st.pop()
                b = st.pop()
                st.append(b + a)
            else:
                return None, unk_symbols
        if len(st) == 1:
            equations = st.pop()
            unk_list = list(unk_symbols.values())
            t = Solver(sym.solve, equations, unk_list)
            t.setDaemon(True)
            t.start()
            t.join(10)
            result = t.get_result()
            return result, unk_symbols
        return None, unk_symbols

    def compute_expression_by_postfix(self, expression):
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None
        return self.compute_postfix_expression(post_exp)

    def compute_expression_by_postfix_multi(self, expression):
        r"""return solves and unknown number list
        """
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None, None
        return self.compute_postfix_expression_multi(post_exp)

    def eval_source(self, test_res, test_tar, num_list, num_stack):
        num_len = len(num_list)
        new_test_res = []
        for symbol in test_res:
            if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
                break
            elif "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                new_test_res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    new_test_res.append(c)
                except:
                    new_test_res=None
                    break
            else:
                new_test_res.append(symbol)
        new_test_tar = []
        for symbol in test_tar:
            if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
                break
            elif "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    return None
                new_test_tar.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    new_test_tar.append(c)
                except:
                    new_test_tar=None
                    break
            else:
                new_test_tar.append(symbol)

        #res_ans = self.compute_expression_by_postfix(new_test_res)
        #tar_ans = self.compute_expression_by_postfix(test_tar, num_list)
        if self.single and self.linear:
            try:
                if abs(self.compute_expression_by_postfix(new_test_res) - self.compute_expression_by_postfix(new_test_tar)) < 1e-4:
                    return True, False, new_test_res, new_test_tar
                else:
                    return False, False, new_test_res, new_test_tar
            except:
                return False, False, new_test_res, new_test_tar
        else:
            try:
                if abs(self.compute_expression_by_postfix_multi(new_test_res) - self.compute_expression_by_postfix_multi(new_test_tar)) < 1e-4:
                    return True, False, new_test_res, new_test_tar
                else:
                    return False, False, new_test_res, new_test_tar
            except:
                return False, False, new_test_res, new_test_tar


class MultiEncDecEvaluator(PostEvaluator,PreEvaluator):
    def __init__(self, symbol2idx, idx2symbol, config):
        super().__init__(symbol2idx, idx2symbol, config)

    def prefix_result(self,test,tar):
        if (self.single and self.linear) != True:  # single but non-linear
            return self.prefix_result_multi(test,tar)
        
        if test is []:
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
    def prefix_result_multi(self,test,tar):
        if test is []:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            test_solves, test_unk = self.compute_prefix_expression_multi(test)
            tar_solves, tar_unk = self.compute_prefix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])
                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
        return False, False, test, tar
    
    def postfix_result(self,test,tar):
        if (self.single and self.linear) != True:  # single but non-linear
            return self.postfix_result_multi(test,tar)
        #test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
        #tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is []:
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
    
    def postfix_result_multi(self,test,tar):
        if test is []:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            test_solves, test_unk = self.compute_postfix_expression_multi(test)
            tar_solves, tar_unk = self.compute_postfix_expression_multi(tar)
            if len(test_unk) != len(tar_unk):
                return False, False, test, tar
            flag = False
            if len(tar_unk) == 1:
                if len(tar_solves) == 1:
                    test_ans = test_solves[list(test_unk.values())[0]]
                    tar_ans = tar_solves[list(tar_unk.values())[0]]
                    if abs(test_ans - tar_ans) < 1e-4:
                        flag = True
                else:
                    flag = True
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        if abs(test_ans[0] - tar_ans[0]) > 1e-4:
                            flag = False
                            break

            else:
                if len(tar_solves) == len(tar_unk):
                    flag = True
                    for tar_x in list(tar_unk.values()):
                        test_ans = test_solves[tar_x]
                        tar_ans = tar_solves[tar_x]
                        if abs(test_ans - tar_ans) > 1e-4:
                            flag = False
                            break
                else:
                    for test_ans, tar_ans in zip(test_solves, tar_solves):
                        try:
                            te_ans = float(test_ans[0])

                        except:
                            te_ans = float(test_ans[1])
                        try:
                            ta_ans = float(tar_ans[0])
                        except:
                            ta_ans = float(tar_ans[1])
                        if abs(te_ans - ta_ans) > 1e-4:
                            flag = False
                            break
            if flag == True:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar
        return False, False, test, tar

    def result(self,test,tar):
        raise NotImplementedError
    def result_multi(self, test, tar):
        raise NotImplementedError
# class SeqEvaluator(AbstractEvaluator):
#     r"""evaluator for normal equation sequnence.
#     """
#     def __init__(self, symbol2idx, idx2symbol, config):
#         super().__init__(symbol2idx, idx2symbol, config)

#     def result(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate single equation.
#         """
#         if (self.single and self.linear) != True:  # single but non-linear
#             return self.result_multi(test_res, test_tar, num_list, num_stack)
        
#         res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         if res_exp == None:
#             return False, False, res_exp, tar_exp
#         if res_exp == tar_exp:
#             return True, True, res_exp, tar_exp
#         try:
#             if abs(self.compute_expression_by_postfix(res_exp) - self.compute_expression_by_postfix(tar_exp)) < 1e-4:
#                 return True, False, tar_exp, tar_exp
#             else:
#                 return False, False, tar_exp, tar_exp
#         except:
#             return False, False, tar_exp, tar_exp

#     def result_multi(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate multiple euqations.
#         """
#         #print(test_tar)# tensor([20,  8, 10, 12,  5, 13, 11,  2])
#         res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         #print(tar_exp)# ['x', '=', '(', '4', '*', '3', ')']
#         if res_exp == None:
#             return False, False, res_exp, tar_exp
#         if res_exp == tar_exp:
#             return True, True, res_exp, tar_exp
#         try:
#             test_solves, test_unk = self.compute_expression_by_postfix_multi(res_exp)
#             tar_solves, tar_unk = self.compute_expression_by_postfix_multi(tar_exp)
#             if len(test_unk) != len(tar_unk):
#                 return False, False, res_exp, tar_exp
#             flag = False
#             if len(tar_unk) == 1:
#                 if len(tar_solves) == 1:
#                     test_ans = test_solves[list(test_unk.values())[0]]
#                     tar_ans = tar_solves[list(tar_unk.values())[0]]
#                     if abs(test_ans - tar_ans) < 1e-4:
#                         flag = True
#                 else:
#                     flag = True
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         if abs(test_ans[0] - tar_ans[0]) > 1e-4:
#                             flag = False
#                             break

#             else:
#                 if len(tar_solves) == len(tar_unk):
#                     flag = True
#                     for tar_x in list(tar_unk.values()):
#                         test_ans = test_solves[tar_x]
#                         tar_ans = tar_solves[tar_x]
#                         if abs(test_ans - tar_ans) > 1e-4:
#                             flag = False
#                             break
#                 else:
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         try:
#                             te_ans = float(test_ans[0])

#                         except:
#                             te_ans = float(test_ans[1])
#                         try:
#                             ta_ans = float(tar_ans[0])
#                         except:
#                             ta_ans = float(tar_ans[1])
#                         if abs(te_ans - ta_ans) > 1e-4:
#                             flag = False
#                             break
#             if flag == True:
#                 return True, False, tar_exp, tar_exp
#             else:
#                 return False, False, tar_exp, tar_exp
#         except:
#             return False, False, tar_exp, tar_exp

#     def out_expression_list(self, test, num_list, num_stack=None):
#         #alphabet="abcdefghijklmnopqrstuvwxyz"
#         num_len = len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
#                 break
#             symbol = self.idx2symbol[i]
#             if "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 res.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     res.append(c)
#                 except:
#                     return None
#             else:
#                 res.append(symbol)
#         if res == []:
#             return None
#         return res

#     def compute_postfix_expression(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b * a)
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
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None
#                 st.append(b**a)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None

#     def compute_postfix_expression_multi(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
#         unk_symbols = {}
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 elif p.isalpha():
#                     if p in unk_symbols:
#                         st.append(unk_symbols[p])
#                     else:
#                         x = sym.symbols(p)
#                         st.append(x)
#                         unk_symbols[p] = x
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b * a)
#             elif p == "/" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if a == 0:
#                     return None, unk_symbols
#                 st.append(b / a)
#             elif p == "-" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b - a)
#             elif p == "^" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None, unk_symbols
#                 st.append(b**a)
#             elif p == "=":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append([sym.Eq(b, a)])
#             elif p == "<BRG>":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             else:
#                 return None, unk_symbols
#         if len(st) == 1:
#             equations = st.pop()
#             unk_list = list(unk_symbols.values())
#             t = Solver(sym.solve, equations, unk_list)
#             t.setDaemon(True)
#             t.start()
#             t.join(10)
#             result = t.get_result()
#             return result, unk_symbols
#         return None, unk_symbols

#     def compute_expression_by_postfix(self, expression):
#         try:
#             post_exp = from_infix_to_postfix(expression)
#         except:
#             return None
#         return self.compute_postfix_expression(post_exp)

#     def compute_expression_by_postfix_multi(self, expression):
#         r"""return solves and unknown number list
#         """
#         try:
#             post_exp = from_infix_to_postfix(expression)
#         except:
#             return None, None
#         return self.compute_postfix_expression_multi(post_exp)

#     def eval_source(self, test_res, test_tar, num_list, num_stack):
#         num_len = len(num_list)
#         new_test_res = []
#         for symbol in test_res:
#             if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
#                 break
#             elif "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 new_test_res.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     new_test_res.append(c)
#                 except:
#                     new_test_res=None
#                     break
#             else:
#                 new_test_res.append(symbol)
#         new_test_tar = []
#         for symbol in test_tar:
#             if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
#                 break
#             elif "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 new_test_tar.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     new_test_tar.append(c)
#                 except:
#                     new_test_tar=None
#                     break
#             else:
#                 new_test_tar.append(symbol)

#         #res_ans = self.compute_expression_by_postfix(new_test_res)
#         #tar_ans = self.compute_expression_by_postfix(test_tar, num_list)
#         if self.single and self.linear:
#             try:
#                 if abs(self.compute_expression_by_postfix(new_test_res) - self.compute_expression_by_postfix(new_test_tar)) < 1e-4:
#                     return True, False, new_test_res, new_test_tar
#                 else:
#                     return False, False, new_test_res, new_test_tar
#             except:
#                 return False, False, new_test_res, new_test_tar
#         else:
#             try:
#                 if abs(self.compute_expression_by_postfix_multi(new_test_res) - self.compute_expression_by_postfix_multi(new_test_tar)) < 1e-4:
#                     return True, False, new_test_res, new_test_tar
#                 else:
#                     return False, False, new_test_res, new_test_tar
#             except:
#                 return False, False, new_test_res, new_test_tar


# class PreEvaluator(AbstractEvaluator):
#     r"""evaluator for prefix equation.
#     """
#     def __init__(self, symbol2idx, idx2symbol, config):
#         super().__init__(symbol2idx, idx2symbol, config)

#     def result(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate single equation.
#         """
#         if (self.single and self.linear) != True:  # single but non-linear
#             return self.result_multi(test_res, test_tar, num_list, num_stack)
#         test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
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

#     def result_multi(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate multiple euqations.
#         """
#         test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         #test = tar
#         #print(test, tar)
#         if test is None:
#             return False, False, test, tar
#         if test == tar:
#             return True, True, test, tar
#         try:
#             test_solves, test_unk = self.compute_prefix_expression_multi(test)
#             tar_solves, tar_unk = self.compute_prefix_expression_multi(tar)
#             if len(test_unk) != len(tar_unk):
#                 return False, False, test, tar
#             flag = False
#             if len(tar_unk) == 1:
#                 if len(tar_solves) == 1:
#                     test_ans = test_solves[list(test_unk.values())[0]]
#                     tar_ans = tar_solves[list(tar_unk.values())[0]]
#                     if abs(test_ans - tar_ans) < 1e-4:
#                         flag = True
#                 else:
#                     flag = True
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         if abs(test_ans[0] - tar_ans[0]) > 1e-4:
#                             flag = False
#                             break

#             else:
#                 if len(tar_solves) == len(tar_unk):
#                     flag = True
#                     for tar_x in list(tar_unk.values()):
#                         test_ans = test_solves[tar_x]
#                         tar_ans = tar_solves[tar_x]
#                         if abs(test_ans - tar_ans) > 1e-4:
#                             flag = False
#                             break
#                 else:
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         try:
#                             te_ans = float(test_ans[0])
#                         except:
#                             te_ans = float(test_ans[1])
#                         try:
#                             ta_ans = float(tar_ans[0])
#                         except:
#                             ta_ans = float(tar_ans[1])
#                         if abs(te_ans - ta_ans) > 1e-4:
#                             flag = False
#                             break
#             if flag == True:
#                 return True, False, test, tar
#             else:
#                 return False, False, test, tar
#         except:
#             return False, False, test, tar
#         return False, False, test, tar

#     def out_expression_list(self, test, num_list, num_stack=None):
#         #alphabet="abcdefghijklmnopqrstuvwxyz"
#         num_len = len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
#                 break
#             symbol = self.idx2symbol[i]
#             if "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 res.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     res.append(c)
#                 except:
#                     return None
#             else:
#                 res.append(symbol)
#         if res == []:
#             return None
#         return res

#     def compute_prefix_expression(self, pre_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         pre_fix = copy.deepcopy(pre_fix)
#         pre_fix.reverse()
#         for p in pre_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
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
#                 st.append(a**b)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None

#     def compute_prefix_expression_multi(self, pre_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
#         unk_symbols = {}
#         pre_fix = copy.deepcopy(pre_fix)
#         pre_fix.reverse()
#         for p in pre_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 elif p.isalpha():
#                     if p in unk_symbols:
#                         st.append(unk_symbols[p])
#                     else:
#                         x = sym.symbols(p)
#                         st.append(x)
#                         unk_symbols[p] = x
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
#                 st.append(a**b)
#             elif p == "=":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append([sym.Eq(a, b)])
#             elif p == "<BRG>":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(a + b)
#             else:
#                 return None
#         if len(st) == 1:
#             equations = st.pop()
#             unk_list = list(unk_symbols.values())
#             t = Solver(sym.solve, equations, unk_list)
#             t.setDaemon(True)
#             t.start()
#             t.join(10)
#             result = t.get_result()
#             return result, unk_symbols
#         return None

#     def eval_source(self, test_res, test_tar, num_list, num_stack=None):
#         raise NotImplementedError


# class PostEvaluator(AbstractEvaluator):
#     r"""evaluator for postfix equation.
#     """
#     def __init__(self, symbol2idx, idx2symbol, config):
#         super().__init__(symbol2idx, idx2symbol, config)

#     def result(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate single equation.
#         """
#         if (self.single and self.linear) != True:  # single but non-linear
#             return self.result_multi(test_res, test_tar, num_list, num_stack)
#         test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
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

#     def result_multi(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate multiple euqations.
#         """
#         test = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         if test is None:
#             return False, False, test, tar
#         if test == tar:
#             return True, True, test, tar
#         try:
#             test_solves, test_unk = self.compute_postfix_expression_multi(test)
#             tar_solves, tar_unk = self.compute_postfix_expression_multi(tar)
#             if len(test_unk) != len(tar_unk):
#                 return False, False, test, tar
#             flag = False
#             if len(tar_unk) == 1:
#                 if len(tar_solves) == 1:
#                     test_ans = test_solves[list(test_unk.values())[0]]
#                     tar_ans = tar_solves[list(tar_unk.values())[0]]
#                     if abs(test_ans - tar_ans) < 1e-4:
#                         flag = True
#                 else:
#                     flag = True
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         if abs(test_ans[0] - tar_ans[0]) > 1e-4:
#                             flag = False
#                             break

#             else:
#                 if len(tar_solves) == len(tar_unk):
#                     flag = True
#                     for tar_x in list(tar_unk.values()):
#                         test_ans = test_solves[tar_x]
#                         tar_ans = tar_solves[tar_x]
#                         if abs(test_ans - tar_ans) > 1e-4:
#                             flag = False
#                             break
#                 else:
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         try:
#                             te_ans = float(test_ans[0])

#                         except:
#                             te_ans = float(test_ans[1])
#                         try:
#                             ta_ans = float(tar_ans[0])
#                         except:
#                             ta_ans = float(tar_ans[1])
#                         if abs(te_ans - ta_ans) > 1e-4:
#                             flag = False
#                             break
#             if flag == True:
#                 return True, False, test, tar
#             else:
#                 return False, False, test, tar
#         except:
#             return False, False, test, tar
#         return False, False, test, tar

#     def out_expression_list(self, test, num_list, num_stack=None):
#         #alphabet="abcdefghijklmnopqrstuvwxyz"
#         num_len = len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
#                 break
#             symbol = self.idx2symbol[i]
#             if "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 res.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     res.append(c)
#                 except:
#                     return None
#             else:
#                 res.append(symbol)
#         if res == []:
#             return None
#         return res

#     def compute_postfix_expression(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
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
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None
#                 st.append(b**a)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None

#     def compute_postfix_expression_multi(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
#         unk_symbols = {}
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 elif p.isalpha():
#                     if p in unk_symbols:
#                         st.append(unk_symbols[p])
#                     else:
#                         x = sym.symbols(p)
#                         st.append(x)
#                         unk_symbols[p] = x
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b * a)
#             elif p == "/" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if a == 0:
#                     return None, unk_symbols
#                 st.append(b / a)
#             elif p == "-" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b - a)
#             elif p == "^" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None, unk_symbols
#                 st.append(b**a)
#             elif p == "=":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append([sym.Eq(b, a)])
#             elif p == "<BRG>":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             else:
#                 return None, unk_symbols
#         if len(st) == 1:
#             equations = st.pop()
#             unk_list = list(unk_symbols.values())
#             t = Solver(sym.solve, equations, unk_list)
#             t.setDaemon(True)
#             t.start()
#             t.join(10)
#             result = t.get_result()
#             return result, unk_symbols
#         return None, unk_symbols

#     def eval_source(self):
#         raise NotImplementedError


# class MultiWayTreeEvaluator(AbstractEvaluator):
#     def __init__(self, symbol2idx, idx2symbol, config):
#         super().__init__(symbol2idx, idx2symbol, config)

#     def result(self, test_res, test_tar, num_list, num_stack):
#         if (self.single and self.linear) != True:  # single but non-linear
#             return self.result_multi(test_res, test_tar, num_list, num_stack)
#         res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         res_exp = copy.deepcopy(tar_exp)
#         if res_exp == None:
#             return False, False, res_exp, tar_exp
#         # if res_exp == tar_exp:
#         #     return True, True, res_exp, tar_exp
#         try:
#             if abs(self.compute_expression_by_postfix(res_exp) - self.compute_expression_by_postfix(tar_exp)) < 1e-4:
#                 return True, False, tar_exp, tar_exp
#             else:
#                 return False, False, tar_exp, tar_exp
#         except:
#             return False, False, tar_exp, tar_exp

#     def result_multi(self, test_res, test_tar, num_list, num_stack):
#         r"""evaluate multiple euqations.
#         """
#         res_exp = self.out_expression_list(test_res, num_list, copy.deepcopy(num_stack))
#         tar_exp = self.out_expression_list(test_tar, num_list, copy.deepcopy(num_stack))
#         if res_exp == None:
#             return False, False, res_exp, tar_exp
#         if res_exp == tar_exp:
#             return True, True, res_exp, tar_exp
#         try:
#             test_solves, test_unk = self.compute_expression_by_postfix_multi(res_exp)
#             tar_solves, tar_unk = self.compute_expression_by_postfix_multi(tar_exp)
#             if len(test_unk) != len(tar_unk):
#                 return False, False, res_exp, tar_exp
#             flag = False
#             if len(tar_unk) == 1:
#                 if len(tar_solves) == 1:
#                     test_ans = test_solves[list(test_unk.values())[0]]
#                     tar_ans = tar_solves[list(tar_unk.values())[0]]
#                     if abs(test_ans - tar_ans) < 1e-4:
#                         flag = True
#                 else:
#                     flag = True
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         if abs(test_ans[0] - tar_ans[0]) > 1e-4:
#                             flag = False
#                             break

#             else:
#                 if len(tar_solves) == len(tar_unk):
#                     flag = True
#                     for tar_x in list(tar_unk.values()):
#                         test_ans = test_solves[tar_x]
#                         tar_ans = tar_solves[tar_x]
#                         if abs(test_ans - tar_ans) > 1e-4:
#                             flag = False
#                             break
#                 else:
#                     for test_ans, tar_ans in zip(test_solves, tar_solves):
#                         try:
#                             te_ans = float(test_ans[0])

#                         except:
#                             te_ans = float(test_ans[1])
#                         try:
#                             ta_ans = float(tar_ans[0])
#                         except:
#                             ta_ans = float(tar_ans[1])
#                         if abs(te_ans - ta_ans) > 1e-4:
#                             flag = False
#                             break
#             if flag == True:
#                 return True, False, tar_exp, tar_exp
#             else:
#                 return False, False, tar_exp, tar_exp
#         except:
#             return False, False, tar_exp, tar_exp
    
#     def out_expression_list(self, test, num_list, num_stack=None):
#         num_len = len(num_list)
#         max_index = len(self.idx2symbol)
#         res = []
#         for i in test:
#             if isinstance(i,list):
#                 sub_res=self.out_expression_list(i,num_list,num_stack)
#                 if sub_res==None:
#                     return None
#                 res.append("(")
#                 res+=sub_res
#                 res.append(")")
#             else:
#                 if i in [self.pad_idx, self.eos_idx, self.sos_idx]:
#                     break
#                 symbol = self.idx2symbol[i]
#                 if "NUM" in symbol:
#                     num_idx = self.mask_list.index(symbol)
#                     if num_idx >= num_len:
#                         return None
#                     res.append(num_list[num_idx])
#                 elif symbol == SpecialTokens.UNK_TOKEN:
#                     try:
#                         pos_list = num_stack.pop()
#                         c = num_list[pos_list[0]]
#                         res.append(c)
#                     except:
#                         return None
#                 else:
#                     res.append(symbol)
#         if res == []:
#             return None
#         return res
    
#     def compute_postfix_expression(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/"]
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b * a)
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
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None
#                 st.append(b**a)
#             else:
#                 return None
#         if len(st) == 1:
#             return st.pop()
#         return None

#     def compute_postfix_expression_multi(self, post_fix):
#         st = list()
#         operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
#         unk_symbols = {}
#         for p in post_fix:
#             if p not in operators:
#                 pos = re.search("\d+\(", p)
#                 if pos:
#                     st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
#                 elif p[-1] == "%":
#                     st.append(float(p[:-1]) / 100)
#                 elif p.isalpha():
#                     if p in unk_symbols:
#                         st.append(unk_symbols[p])
#                     else:
#                         x = sym.symbols(p)
#                         st.append(x)
#                         unk_symbols[p] = x
#                 else:
#                     st.append(eval(p))
#             elif p == "+" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             elif p == "*" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b * a)
#             elif p == "/" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if a == 0:
#                     return None, unk_symbols
#                 st.append(b / a)
#             elif p == "-" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b - a)
#             elif p == "^" and len(st) > 1:
#                 a = st.pop()
#                 b = st.pop()
#                 if float(a) != 2.0 and float(a) != 3.0:
#                     return None, unk_symbols
#                 st.append(b**a)
#             elif p == "=":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append([sym.Eq(b, a)])
#             elif p == "<BRG>":
#                 a = st.pop()
#                 b = st.pop()
#                 st.append(b + a)
#             else:
#                 return None, unk_symbols
#         if len(st) == 1:
#             equations = st.pop()
#             unk_list = list(unk_symbols.values())
#             t = Solver(sym.solve, equations, unk_list)
#             t.setDaemon(True)
#             t.start()
#             t.join(10)
#             result = t.get_result()
#             return result, unk_symbols
#         return None, unk_symbols

#     def compute_expression_by_postfix(self, expression):
#         try:
#             post_exp = from_infix_to_postfix(expression)
#         except:
#             return None
#         return self.compute_postfix_expression(post_exp)

#     def compute_expression_by_postfix_multi(self, expression):
#         r"""return solves and unknown number list
#         """
#         try:
#             post_exp = from_infix_to_postfix(expression)
#         except:
#             return None, None
#         return self.compute_postfix_expression_multi(post_exp)

#     def eval_source(self, test_res, test_tar, num_list, num_stack):
#         num_len = len(num_list)
#         new_test_res = []
#         for symbol in test_res:
#             if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
#                 break
#             elif "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 new_test_res.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     new_test_res.append(c)
#                 except:
#                     new_test_res=None
#                     break
#             else:
#                 new_test_res.append(symbol)
#         new_test_tar = []
#         for symbol in test_tar:
#             if symbol in [SpecialTokens.PAD_TOKEN,SpecialTokens.EOS_TOKEN]:
#                 break
#             elif "NUM" in symbol:
#                 num_idx = self.mask_list.index(symbol)
#                 if num_idx >= num_len:
#                     return None
#                 new_test_tar.append(num_list[num_idx])
#             elif symbol == SpecialTokens.UNK_TOKEN:
#                 try:
#                     pos_list = num_stack.pop()
#                     c = num_list[pos_list[0]]
#                     new_test_tar.append(c)
#                 except:
#                     new_test_tar=None
#                     break
#             else:
#                 new_test_tar.append(symbol)

#         #res_ans = self.compute_expression_by_postfix(new_test_res)
#         #tar_ans = self.compute_expression_by_postfix(test_tar, num_list)
#         if self.single and self.linear:
#             try:
#                 if abs(self.compute_expression_by_postfix(new_test_res) - self.compute_expression_by_postfix(new_test_tar)) < 1e-4:
#                     return True, False, new_test_res, new_test_tar
#                 else:
#                     return False, False, new_test_res, new_test_tar
#             except:
#                 return False, False, new_test_res, new_test_tar
#         else:
#             try:
#                 if abs(self.compute_expression_by_postfix_multi(new_test_res) - self.compute_expression_by_postfix_multi(new_test_tar)) < 1e-4:
#                     return True, False, new_test_res, new_test_tar
#                 else:
#                     return False, False, new_test_res, new_test_tar
#             except:
#                 return False, False, new_test_res, new_test_tar
