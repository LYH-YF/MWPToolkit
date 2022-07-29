# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 19:19:55
# @File: evaluator.py


import copy
import re
import threading
from typing import Type, Union

import sympy as sym

from mwptoolkit.config.configuration import Config
from mwptoolkit.utils.enum_type import SpecialTokens, OPERATORS, NumMask, MaskSymbol, FixType
from mwptoolkit.utils.preprocess_tools import from_infix_to_postfix


class Solver(threading.Thread):
    r"""time-limited equation-solving mechanism based threading.
    """
    def __init__(self, func, equations, unk_symbol):
        super(Solver, self).__init__()
        """
        Args:
            func (function): a function to solve equations.

            equations (list): list of expressions.

            unk_symbol (list): list of unknown symbols.
        """
        self.func = func
        self.equations = equations
        self.unk_symbol = unk_symbol

    def run(self):
        """run equation solving process
        """
        try:
            self.result = self.func(self.equations, self.unk_symbol)
        except:
            self.result = None

    def get_result(self):
        """return the result
        """
        try:
            return self.result
        except:
            return None


class AbstractEvaluator(object):
    """abstract evaluator
    """
    def __init__(self, config):
        super().__init__()
        self.share_vocab = config["share_vocab"]
        self.mask_symbol = config["mask_symbol"]
        self.task_type = config["task_type"]
        self.single = config["single"]
        self.linear = config["linear"]

    def result(self):
        raise NotImplementedError

    def result_multi(self):
        raise NotImplementedError


class InfixEvaluator(AbstractEvaluator):
    r"""evaluator for infix equation sequnence.
    """
    def __init__(self, config):
        super().__init__(config)

    def result(self, test_exp, tar_exp):
        """evaluate single equation.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): iist of target expression.

        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test_exp, tar_exp)

        if test_exp == []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_expression_by_postfix(test_exp) - self._compute_expression_by_postfix(tar_exp)) < 1e-4:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def result_multi(self, test_exp, tar_exp):
        """evaluate multiple euqations.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if test_exp == []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_expression_by_postfix_multi(test_exp)
            tar_solves, tar_unk = self._compute_expression_by_postfix_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def _compute_postfix_expression(self, post_fix):
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

    def _compute_postfix_expression_multi(self, post_fix):
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

    def _compute_expression_by_postfix(self, expression):
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None
        return self._compute_postfix_expression(post_exp)

    def _compute_expression_by_postfix_multi(self, expression):
        r"""return solves and unknown number list
        """
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None, None
        return self._compute_postfix_expression_multi(post_exp)


class PrefixEvaluator(AbstractEvaluator):
    r"""evaluator for prefix equation.
    """
    def __init__(self, config):
        super().__init__(config)

    def result(self, test_exp, tar_exp):
        """evaluate single equation.

        Args:
            test_exp (list): list of test expression.

            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test_exp, tar_exp)
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_prefix_expression(test_exp) - self._compute_prefix_expression(tar_exp)) < 1e-4:
                return True, False, test_exp, tar_exp
            else:
                return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp

    def result_multi(self, test_exp, tar_exp):
        """evaluate multiple euqations.

        Args:
            test_exp (list): list of test expression.

            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.

            equ_ac (bool): the correctness of test expression compared to target expression.

            test_exp (list): list of test expression.

            tar_exp (list): list of target expression.
        """
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_prefix_expression_multi(test_exp)
            tar_solves, tar_unk = self._compute_prefix_expression_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                return True, False, test_exp, tar_exp
            else:
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp
        return False, False, test_exp, tar_exp

    def _compute_prefix_expression(self, pre_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        pre_fix_ = copy.deepcopy(pre_fix)
        pre_fix_.reverse()
        for p in pre_fix_:
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

    def _compute_prefix_expression_multi(self, pre_fix):
        st = list()
        operators = ["+", "-", "^", "*", "/", "=", "<BRG>"]
        unk_symbols = {}
        pre_fix_ = copy.deepcopy(pre_fix)
        pre_fix_.reverse()
        for p in pre_fix_:
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


class PostfixEvaluator(AbstractEvaluator):
    r"""evaluator for postfix equation.
    """
    def __init__(self, config):
        super().__init__(config)

    def result(self, test_exp, tar_exp):
        """evaluate single equation.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test_exp, tar_exp)
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_postfix_expression(test_exp) - self._compute_postfix_expression(tar_exp)) < 1e-4:
                return True, False, test_exp, tar_exp
            else:
                return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp

    def result_multi(self, test_exp, tar_exp):
        """evaluate multiple euqations.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_postfix_expression_multi(test_exp)
            tar_solves, tar_unk = self._compute_postfix_expression_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                return True, False, test_exp, tar_exp
            else:
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp
        return False, False, test_exp, tar_exp

    def _compute_postfix_expression(self, post_fix):
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

    def _compute_postfix_expression_multi(self, post_fix):
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
    def __init__(self, config):
        super().__init__(config)

    def result(self, test_exp, tar_exp):
        """evaluate single equation.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.result_multi(test_exp, tar_exp)
        if test_exp == []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_expression_by_postfix(test_exp) - self._compute_expression_by_postfix(tar_exp)) < 1e-4:
                return True, False, tar_exp, tar_exp
            else:
                return False, False, tar_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def result_multi(self, test_exp, tar_exp):
        r"""evaluate multiple euqations.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if test_exp == []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_expression_by_postfix_multi(test_exp)
            tar_solves, tar_unk = self._compute_expression_by_postfix_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, tar_exp, tar_exp

    def _compute_postfix_expression(self, post_fix):
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

    def _compute_postfix_expression_multi(self, post_fix):
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

    def _compute_expression_by_postfix(self, expression):
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None
        return self._compute_postfix_expression(post_exp)

    def _compute_expression_by_postfix_multi(self, expression):
        r"""return solves and unknown number list
        """
        try:
            post_exp = from_infix_to_postfix(expression)
        except:
            return None, None
        return self._compute_postfix_expression_multi(post_exp)


class MultiEncDecEvaluator(PostfixEvaluator, PrefixEvaluator):
    r"""evaluator for deep-learning model MultiE&D.
    """
    def __init__(self, config):
        super().__init__(config)

    def prefix_result(self, test_exp, tar_exp):
        """evaluate single prefix equation.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.prefix_result_multi(test_exp, tar_exp)

        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_prefix_expression(test_exp) - self._compute_prefix_expression(tar_exp)) < 1e-4:
                return True, False, test_exp, tar_exp
            else:
                return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp

    def prefix_result_multi(self, test_exp, tar_exp):
        """evaluate multiple prefix euqations.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_prefix_expression_multi(test_exp)
            tar_solves, tar_unk = self._compute_prefix_expression_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                return True, False, test_exp, tar_exp
            else:
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp
        return False, False, test_exp, tar_exp

    def postfix_result(self, test_exp, tar_exp):
        """evaluate single postfix equation.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))

            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if (self.single and self.linear) != True:  # single but non-linear
            return self.postfix_result_multi(test_exp, tar_exp)
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            if abs(self._compute_postfix_expression(test_exp) - self._compute_postfix_expression(tar_exp)) < 1e-4:
                return True, False, test_exp, tar_exp
            else:
                return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp

    def postfix_result_multi(self, test_exp, tar_exp):
        """evaluate multiple postfix euqations.

        Args:
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.

        Returns:
            (tuple(bool,bool,list,list))
            
            val_ac (bool): the correctness of test expression answer compared to target expression answer.
            
            equ_ac (bool): the correctness of test expression compared to target expression.
            
            test_exp (list): list of test expression.
            
            tar_exp (list): list of target expression.
        """
        if test_exp is []:
            return False, False, test_exp, tar_exp
        if test_exp == tar_exp:
            return True, True, test_exp, tar_exp
        try:
            test_solves, test_unk = self._compute_postfix_expression_multi(test_exp)
            tar_solves, tar_unk = self._compute_postfix_expression_multi(tar_exp)
            if len(test_unk) != len(tar_unk):
                return False, False, test_exp, tar_exp
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
                return True, False, test_exp, tar_exp
            else:
                flag = True
                test_solves_list = list(test_solves.values())
                target_solvers_list = list(tar_solves.values())
                t1 = sorted(test_solves_list)
                t2 = sorted(target_solvers_list)
                for v1, v2 in zip(t1, t2):
                    if abs(v1 - v2) > 1e-4:
                        flag = False
                        break
                if flag:
                    return True, False, test_exp, tar_exp
                else:
                    return False, False, test_exp, tar_exp
        except:
            return False, False, test_exp, tar_exp
        return False, False, test_exp, tar_exp

    def result(self, test_exp, tar_exp):
        raise NotImplementedError

    def result_multi(self, test_exp, tar_exp):
        raise NotImplementedError


def get_evaluator(config):
    """build evaluator

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Evaluator: Constructed evaluator.
    """
    if config["equation_fix"] == FixType.Prefix:
        evaluator = PrefixEvaluator(config)
    elif config["equation_fix"] == FixType.Nonfix or config["equation_fix"] == FixType.Infix:
        evaluator = InfixEvaluator(config)
    elif config["equation_fix"] == FixType.Postfix:
        evaluator = PostfixEvaluator(config)
    elif config["equation_fix"] == FixType.MultiWayTree:
        evaluator = MultiWayTreeEvaluator(config)
    else:
        raise NotImplementedError

    if config['model'].lower() in ['multiencdec']:
        evaluator = MultiEncDecEvaluator(config)

    return evaluator


def get_evaluator_module(config: Config) -> Type[Union[PrefixEvaluator,InfixEvaluator,PostfixEvaluator,MultiWayTreeEvaluator,AbstractEvaluator,MultiEncDecEvaluator]]:
    """return a evaluator module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: evaluator module
    """
    if config["equation_fix"] == FixType.Prefix:
        evaluator_module = PrefixEvaluator
    elif config["equation_fix"] == FixType.Nonfix or config["equation_fix"] == FixType.Infix:
        evaluator_module = InfixEvaluator
    elif config["equation_fix"] == FixType.Postfix:
        evaluator_module = PostfixEvaluator
    elif config["equation_fix"] == FixType.MultiWayTree:
        evaluator_module = MultiWayTreeEvaluator
    else:
        evaluator_module = AbstractEvaluator

    if config['model'].lower() in ['multiencdec']:
        evaluator_module = MultiEncDecEvaluator

    return evaluator_module

