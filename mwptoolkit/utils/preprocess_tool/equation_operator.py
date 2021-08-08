import random
from copy import deepcopy

from mwptoolkit.utils.enum_type import SpecialTokens,NumMask,EPT

def from_infix_to_postfix(expression):
    r"""convert infix equation to postfix equation
    """
    st = list()
    res = list()
    priority = {"<BRG>": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res


def from_infix_to_prefix(expression):
    r"""convert infix equation to prefix equation
    """
    st = list()
    res = list()
    priority = {"<BRG>": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def from_prefix_to_postfix(expression):
    r"""convert prefix equation to postfix equation
    """
    st = list()
    expression = deepcopy(expression)
    expression.reverse()
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n1 = st.pop()
            n2 = st.pop()
            st.append(n1 + n2 + [symbol])
    res = st.pop()
    return res


def from_postfix_to_prefix(expression):
    r"""convert postfix equation to prefix equation
    """
    st = list()
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n1 = st.pop()  #2
            n2 = st.pop()  #3
            st.append([symbol] + n2 + n1)
    res = st.pop()
    return res


def from_prefix_to_infix(expression):
    r"""convert prefix equation to infix equation
    """
    st = list()
    last_op = []
    priority = {"<BRG>": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
    expression = deepcopy(expression)
    expression.reverse()
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n_left = st.pop()
            n_right = st.pop()
            left_first = False
            right_first = False
            if len(n_left) > 1 and priority[last_op.pop()] < priority[symbol]:
                left_first = True
            if len(n_right) > 1 and priority[last_op.pop()] < priority[symbol]:
                right_first = True
            if left_first:
                n_left = ['('] + n_left + [')']
            if right_first:
                n_right = ['('] + n_right + [')']
            st.append(n_left + [symbol] + n_right)
            last_op.append(symbol)
    res = st.pop()
    return res


def from_postfix_to_infix(expression):
    r"""convert postfix equation to infix equation
    """
    st = list()
    last_op = []
    priority = {"<BRG>": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n_right = st.pop()
            n_left = st.pop()
            left_first = False
            right_first = False
            if len(n_right) > 1 and priority[last_op.pop()] < priority[symbol]:
                right_first = True
            if len(n_left) > 1 and priority[last_op.pop()] < priority[symbol]:
                left_first = True
            if left_first:
                n_left = ['('] + n_left + [')']
            if right_first:
                n_right = ['('] + n_right + [')']
            st.append(n_left + [symbol] + n_right)
            last_op.append(symbol)
    res = st.pop()
    return res


def from_infix_to_multi_way_tree(expression):
    res = []
    st = []
    level = 0
    for e in expression:
        if e in ['(', '[']:
            level += 1
            st.append(e)
        elif e in [')', ']']:
            level -= 1
            st.append(e)
            if level == 0:
                sub_res = from_infix_to_multi_way_tree(st[1:-1])
                res.append(sub_res)
                st = []
        else:
            if level != 0:
                st.append(e)
            else:
                res.append(e)
    return res


def postfix_parser(equation, memory: list) -> int:
    """
    Read Op-token postfix equation and transform it into Expression-token sequence.

    :param List[Union[str,Tuple[str,Any]]] equation:
        List of op-tokens to be parsed into a Expression-token sequence
        Item of this list should be either
        - an operator string
        - a tuple of (operand source, operand value)
    :param list memory:
        List where previous execution results of expressions are stored
    :rtype: int
    :return:
        Size of stack after processing. Value 1 means parsing was done without any free expression.
    """
    stack = []

    for tok in equation:
        if tok in EPT.OPERATORS:
            # If token is an operator, form expression and push it into the memory and stack.
            op = EPT.OPERATORS[tok]
            arity = op['arity']

            # Retrieve arguments
            args = stack[-arity:]
            stack = stack[:-arity]

            # Store the result with index where the expression stored
            stack.append((EPT.ARG_MEM, len(memory)))
            # Store the expression into the memory.
            memory.append((tok, args))
        else:
            # Push an operand before meet an operator
            stack.append(tok)

    return len(stack)


def operator_mask(expression):
    template = []
    for symbol in expression:
        if isinstance(symbol, list):
            sub_temp = operator_mask(symbol)
            template.append(sub_temp)
        elif symbol in ["+", "-", "*", "/", "^", "=", "<BRG>"]:
            template.append(SpecialTokens.OPT_TOKEN)
        else:
            template.append(symbol)
    return template


def trans_symbol_2_number(equ_list, num_list):
    symbol_list = NumMask.number
    new_equ_list = []
    for symbol in equ_list:
        if 'NUM' in symbol:
            index = symbol_list.index(symbol)
            new_equ_list.append(str(num_list[index]))
        else:
            new_equ_list.append(symbol)
    return new_equ_list


def EN_rule1_stat(datas, sample_k=100):
    rule_1 = []
    for data in datas:
        temp_data = data
        equ_list = data["equation"]
        rule_1.append(equ_list)
    samples = random.sample(range(10, 100), k=sample_k)
    random.shuffle(samples)
    ans_dict = {}
    for equ_list in rule_1:
        new_equ = trans_symbol_2_number(equ_list, samples)
        new_equ = ''.join(new_equ)
        new_equ = new_equ.replace("^", "**", 10)
        new_equ = new_equ.replace("[", "(", 10)
        new_equ = new_equ.replace("]", ")", 10)
        try:
            ans = eval(new_equ)
        except:
            ans = float("inf")
        try:
            ans_dict[ans].append(equ_list)
        except:
            ans_dict[ans] = []
            ans_dict[ans].append(equ_list)
    class_list = []
    for k, v in ans_dict.items():
        class_list.append(v)
    for i in range(50):
        samples = random.sample(range(10, 100), k=sample_k)
        random.shuffle(samples)
        class_copy = deepcopy(class_list)
        class_list = []
        for equ_lists in class_copy:
            ans_dict = {}
            for equ_list in equ_lists:
                new_equ = trans_symbol_2_number(equ_list, samples)
                new_equ = ''.join(new_equ)
                new_equ = new_equ.replace("^", "**", 10)
                new_equ = new_equ.replace("[", "(", 10)
                new_equ = new_equ.replace("]", ")", 10)
                try:
                    ans = eval(new_equ)
                except:
                    ans = float("inf")
                try:
                    ans_dict[ans].append(equ_list)
                except:
                    ans_dict[ans] = []
                    ans_dict[ans].append(equ_list)
            for k, v in ans_dict.items():
                class_list.append(v)
    class_copy = deepcopy(class_list)
    class_list = []
    for equ_lists in class_copy:
        class_list_temp = []
        for equ_list in equ_lists:
            if equ_list not in class_list_temp:
                class_list_temp.append(equ_list)
            class_list_temp = sorted(class_list_temp, key=lambda x: len(x), reverse=False)
        class_list.append(class_list_temp)
    return class_list


def EN_rule2(equ_list):
    new_list = []
    i = 0
    while i < len(equ_list):
        if (i + 4) < len(equ_list) and ('NUM' in equ_list[i] or equ_list[i].isalpha()) and '+' in equ_list[i + 1] and (
                'NUM' in equ_list[i + 2] or equ_list[i + 2].isalpha()) and '+' in equ_list[i + 3] and ('NUM' in equ_list[i + 4] or equ_list[i + 4].isalpha()):
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 5 < len(equ_list) and equ_list[i + 5] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            temp = [equ_list[i], equ_list[i + 2], equ_list[i + 4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1] + ['+'] + sort_temp[1:2] + ['+'] + sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i + 4) < len(equ_list) and ('NUM' in equ_list[i] or equ_list[i].isalpha()) and '*' in equ_list[i + 1] and (
                'NUM' in equ_list[i + 2] or equ_list[i + 2].isalpha()) and '*' in equ_list[i + 3] and ('NUM' in equ_list[i + 4] or equ_list[i + 4].isalpha()):
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 5 < len(equ_list) and equ_list[i + 5] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            temp = [equ_list[i], equ_list[i + 2], equ_list[i + 4]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1] + ['*'] + sort_temp[1:2] + ['*'] + sort_temp[2:3]
            new_list += new_temp
            i += 5
        elif (i + 2) < len(equ_list) and ('NUM' in equ_list[i] or equ_list[i].isalpha()) and '+' in equ_list[i + 1] and ('NUM' in equ_list[i + 2] or equ_list[i + 2].isalpha()):
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 3 < len(equ_list) and equ_list[i + 3] in ['/', '-', '*']:
                new_list.append(equ_list[i])
                i += 1
                continue
            temp = [equ_list[i], equ_list[i + 2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1] + ['+'] + sort_temp[1:2]
            new_list += new_temp
            i += 3
        elif (i + 2) < len(equ_list) and ('NUM' in equ_list[i] or equ_list[i].isalpha()) and '*' in equ_list[i + 1] and ('NUM' in equ_list[i + 2] or equ_list[i + 2].isalpha()):
            if i - 1 >= 0 and equ_list[i - 1] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            if i + 3 < len(equ_list) and equ_list[i + 3] in ['/', '-']:
                new_list.append(equ_list[i])
                i += 1
                continue
            temp = [equ_list[i], equ_list[i + 2]]
            sort_temp = sorted(temp)
            new_temp = sort_temp[0:1] + ['*'] + sort_temp[1:2]
            new_list += new_temp
            i += 3
        else:
            new_list.append(equ_list[i])
            i += 1
    return new_list
