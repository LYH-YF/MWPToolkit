import re
from copy import deepcopy
from collections import OrderedDict
from fractions import Fraction
import stanza

from mwptoolkit.utils.enum_type import MaskSymbol, NumMask, SpecialTokens


def split_number(text_list):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    new_text = []
    for s in text_list:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            num = s[pos.start():pos.end()]
            new_text.append(num)
            if pos.end() < len(s):
                new_text.append(s[pos.end():])
        else:
            new_text.append(s)
    return new_text


def joint_number(text_list):
    new_list = []
    i = 0
    while i < len(text_list):
        if text_list[i] == '(' and i + 4 < len(text_list) and text_list[i + 4] == ')':
            sub = ''.join(text_list[i:i + 5])
            new_list.append(sub)
            i = i + 5
        else:
            new_list.append(text_list[i])
            i += 1
    return new_list


def joint_number_(text_list):  #match longer fraction such as ( 1 / 1000000 )
    new_list = []
    i = 0
    while i < len(text_list):
        if text_list[i] == '(':
            try:
                j = text_list[i:].index(')')
                if i + 1 == i + j:
                    j = None
                if "(" in text_list[i + 1:i + j + 1]:
                    j = None
            except:
                j = None
            if j:
                stack = []
                flag = True
                idx = 0
                for temp_idx, word in enumerate(text_list[i:i + j + 1]):
                    if word in ["(", ")", "/"] or word.isdigit():
                        stack.append(word)
                        idx = temp_idx
                    else:
                        flag = False
                        break
                if flag:
                    number = ''.join(stack)
                    new_list.append(number)
                else:
                    for word in stack:
                        new_list.append(word)
                i += idx + 1
            else:
                new_list.append(text_list[i])
                i += 1
        else:
            new_list.append(text_list[i])
            i += 1
    return new_list


def search_number(seq, equ):
    for idx, symbol in enumerate(equ):
        if symbol in seq:
            continue
        else:
            try:
                number = eval(symbol)
                if number in seq:
                    equ[idx] = str(number)
                else:
                    continue
            except:
                continue
    return equ


def seg_and_tag_(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                res.append(nums[number])
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        res.append(ss)
    return res


def seg_and_tag_ape200k(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_ape200k(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_ape200k(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_ape200k(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                res.append(nums[number])
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_ape200k(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        res.append(ss)
    return res


def seg_and_tag_math23k(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-(([(]\d+\.\d+[)])|([(]\d+/\d+[)]))", st)  #search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        try:
            st_num = str(eval(st[p_start:p_end]))
        except:  # % in number
            st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_math23k(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_math23k(st[p_end:], nums_fraction, nums)
            return res

    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_math23k(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                res.append(nums[number])
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_math23k(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        res.append(ss)
    return res


def seg_and_tag_mawps(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_mawps(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-(\d+\.\d+)", st)  #search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        try:
            st_num = str(eval(st[p_start:p_end]))
        except:  # % in number
            st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:], nums_fraction, nums)
        return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  #search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        try:
            st_num = str(eval(st[p_start:p_end]))
        except:  # % in number
            st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:], nums_fraction, nums)
        return res
    pos_st = re.search("<BRG>", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        res.append(st[p_start:p_end])
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res


def number_transfer_(data, mask_type="NUM", min_generate_keep=0):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        equations = d["equation"][2:]
        if '千' in equations:
            equations = equations[:equations.index('千')]

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    #nums.append(s[pos.start():pos.end()])
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            input_seq.append(nums[s[pos.start():pos.end()]])
                    except:
                        nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    finally:
                        num_list.append(s[pos.start():pos.end()])
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==133813:
        #     print(1)
        out_seq = seg_and_tag_(equations, nums_fraction, nums)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)
        assert len(num_list) == len(num_pos)

        #copy data
        # if d["id"]=="8883":
        #     print(1)
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums


def number_transfer_math23k(data, mask_type="NUM", min_generate_keep=0):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        equations = d["equation"][2:]
        if '千' in equations:
            equations = equations[:equations.index('千')]

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    #nums.append(s[pos.start():pos.end()])
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            input_seq.append(nums[s[pos.start():pos.end()]])
                    except:
                        nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    finally:
                        num_list.append(s[pos.start():pos.end()])
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]=='23100':
        #     print(1)
        out_seq = seg_and_tag_math23k(equations, nums_fraction, nums)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)
        assert len(num_list) == len(num_pos)
        #copy data
        # if d["id"]=="8883":
        #     print(1)
        new_data = d
        new_data["question"] = input_seq
        new_data["ques source 1"] = source
        new_data["equation"] = out_seq
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums


def number_transfer_ape200k(data, mask_type="NUM", min_generate_keep=0):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        seg = joint_number_(seg)
        equations = d["equation"]
        if "x=" == equations[:2] or "X=" == equations[:2]:
            equations = equations[2:]
        if '千' in equations:
            equations = equations[:equations.index('千')]

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    #nums.append(s[pos.start():pos.end()])
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            input_seq.append(nums[s[pos.start():pos.end()]])
                    except:
                        nums[s[pos.start():pos.end()]] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_ape200k(equations, nums_fraction, nums)
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums


def num_transfer_mawps(data, mask_type="number", min_generate_keep=0):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["segmented_text"].split(" ")
        equations = d["equation"]

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    number = str(eval(s[pos.start():pos.end()]))
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==76:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums


def num_transfer_multi(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        # if d["id"]==2022:
        #     print(1)
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    try:
                        number = str(eval(s[pos.start():pos.end()]))
                    except:  # "%" in number
                        number = s[pos.start():pos.end()]
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==76:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def num_transfer_alg514(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        # if d["id"]==2022:
        #     print(1)
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    try:
                        number = str(eval(s[pos.start():pos.end()]))
                    except:  # "%" in number
                        number = s[pos.start():pos.end()]
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==6666:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        if num_list == []:
            new_data["number list"] = ["-inf"]
            new_data["number position"] = [-1]
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def num_transfer_draw(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        # if d["id"]==2022:
        #     print(1)
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
            elif re.match(r"\.\d+", word):
                new_word = "0" + word
                seg[idx] = new_word

        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    try:
                        number = str(eval(s[pos.start():pos.end()]))
                    except:  # "%" in number
                        number = s[pos.start():pos.end()]
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==34492:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            # if s=="18.0" or s=="12.0" or s=="162.0":
            #     print(1)
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j and j not in ["NUMBERS","NUMBER"]:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        if re.match("\d+\(\d+/\d+\)", num_str):  # match fraction like '5(3/4)'
                            idx = num_str.index('(')
                            a = num_str[:idx]
                            b = num_str[idx:]
                        if re.match("\(\d+/\d+\)\d+", num_str):  # match fraction like '(3/4)5'
                            idx = num_str.index(')')
                            a = num_str[:idx + 1]
                            b = num_str[idx + 1:]
                        num = str(eval(a) + eval(b))
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        if num_list == []:
            new_data["number list"] = ["-inf"]
            new_data["number position"] = [-1]
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def num_transfer_hmwp(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        nums = OrderedDict()
        input_seq = []
        # text=d["original_text"]
        # pos=re.search(r'\d+\s\d+',text)
        # while(pos):
        #     start=pos.start()
        #     end=pos.end()
        #     number=text[start:end]
        #     number=''.join(number.split(" "))
        #     text=text[:start]+number+text[end:]
        #     pos=re.search(r'\d+\s\d+',text)
        # seg = text.split(" ")
        seg = d["original_text"].split(" ")
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    try:
                        number = str(eval(s[pos.start():pos.end()]))
                    except:  # "%" in number
                        number = s[pos.start():pos.end()]
                    try:
                        if mask_type == "NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number] = equ_mask_list[equ_idx]
                            sent_idx = (sent_idx + 1) % len(sent_mask_list)
                            equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number] = equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx = (equ_idx + 1) % len(equ_mask_list)
                        sent_idx = (sent_idx + 1) % len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==76:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        num_list = list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list = list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)
        source = deepcopy(input_seq)
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_idx = equ_mask_list.index(j)
                num_str = num_list[num_idx]
                if '%' in num_str:
                    num = str(eval(num_str[:-1] + '/100'))
                else:
                    try:
                        num = str(eval(num_str))
                    except:
                        num = num_str
                    num_list[num_idx] = num
                source[i] = num
        source = ' '.join(source)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["ques source 1"] = source
        new_data["number list"] = num_list
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol



def get_group_nums(datas, language):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
    new_datas = []
    for idx, data in enumerate(datas):
        group_nums = []
        num_pos = data["number position"]
        sent_len = len(data["question"])
        doc = nlp(data["ques source 1"])
        token_list = doc.to_dict()[0]
        for n_pos in num_pos:
            pos_stack = []
            group_num = []
            pos_stack.append([n_pos, token_list[n_pos]["deprel"]])
            head_pos = token_list[n_pos]['head']
            for idx, x in enumerate(token_list):
                if x['head'] == head_pos and n_pos != idx:
                    deprel = x["deprel"]
                    pos_stack.append([idx, deprel])
            while pos_stack:
                pos_dep = pos_stack.pop(0)
                pos = pos_dep[0]
                dep = pos_dep[1]
                head_pos = token_list[pos]['head'] - 1
                upos = token_list[pos]['upos']
                if upos not in ['NOUN', 'NUM', 'ADJ', 'VERB', 'DET', 'SYM']:
                    continue
                elif upos == 'NOUN' and dep not in ['compound', 'nsubj:pass', 'nsubj', 'compound']:
                    continue
                elif upos == 'VERB' and dep not in ['conj', 'root']:
                    continue
                elif upos == 'ADJ' and dep not in ['amod']:
                    continue
                elif upos == 'DET' and dep not in ['advmod']:
                    continue
                elif upos == 'SYM' and dep not in ['obl']:
                    continue
                else:
                    group_num.append(pos)
                if head_pos >= 0:
                    head_dep = token_list[head_pos]['deprel']
                    if [head_pos, head_dep] in pos_stack:
                        pass
                    else:
                        pos_stack.append([head_pos, head_dep])
            if group_num == []:
                group_num.append(n_pos)
            if len(group_num) == 1:
                if n_pos - 1 >= 0:
                    group_num.append(n_pos - 1)
                if n_pos + 1 <= sent_len:
                    group_num.append(n_pos + 1)
            group_nums.append(group_num)
        #datas[idx]["group nums"]=group_nums
        data["group nums"] = group_nums
        new_datas.append(data)

    return new_datas


def operator_mask(expression):
    template = []
    for symbol in expression:
        if symbol in ["+", "-", "*", "/", "^", "=", "<BRG>"]:
            template.append(SpecialTokens.OPT_TOKEN)
        else:
            template.append(symbol)
    return template


def from_infix_to_postfix(expression):
    r"""postfix for expression
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
    r"""prefix for expression
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


def num_transfer_draw_(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)
        num_list_ = d["number list"]
        num_pos_ = d["number_position"]
        num_list = []
        num_pos = []
        for num, pos in zip(num_list_, num_pos_):
            if num in num_list:
                continue
            else:
                num_list.append(num)
                num_pos.append(pos)
        idx = 0
        for num in num_list:
            if num in nums:
                continue
            nums[num] = equ_mask_list[idx]
            idx = (idx + 1) % len(equ_mask_list)

        for idx, s in enumerate(seg):
            if idx in num_pos:
                num_idx = num_pos.index(idx)
                try:
                    if abs(eval(seg[idx]) - eval(num_list[num_idx])) < 1e-5:
                        #seg[idx]=nums[num_list[num_idx]]
                        input_seq.append(nums[num_list[num_idx]])
                    else:
                        input_seq.append(s)
                except:
                    input_seq.append(s)
            else:
                input_seq.append(s)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==6666:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def num_transfer_alg514_(data, mask_type="number", min_generate_keep=0, equ_split_symbol=";"):
    '''transfer num process

    Args:
        data: list.
        mask_type: str | default 'NUM', the way to mask num, optinal['NUM', 'alphabet', 'number'].
        min_generate_keep: int | default 5, the number to control if the numbers of equations will be kept as generating number.

    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    max_equ__len = {}
    unk_symbol = []
    for d in data:
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)
        num_list_ = d["number list"]
        num_pos_ = d["number_position"]
        num_list = []
        num_pos = []
        for num, pos in zip(num_list_, num_pos_):
            if num in num_list:
                continue
            else:
                num_list.append(num)
                num_pos.append(pos)
        idx = 0
        for num in num_list:
            if num in nums:
                continue
            nums[num] = equ_mask_list[idx]
            idx = (idx + 1) % len(equ_mask_list)

        for idx, s in enumerate(seg):
            if idx in num_pos:
                num_idx = num_pos.index(idx)
                try:
                    if abs(eval(seg[idx]) - eval(num_list[num_idx])) < 1e-5:
                        #seg[idx]=nums[num_list[num_idx]]
                        input_seq.append(nums[num_list[num_idx]])
                    else:
                        input_seq.append(s)
                except:
                    input_seq.append(s)
            else:
                input_seq.append(s)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            # if re.search("-\d+|(-\d+\.\d+)",num):
            #     nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # if d["id"]==6666:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)
        # try:
        #     max_equ__len[len(out_seq)]+=1
        # except:
        #     max_equ__len[len(out_seq)]=1
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        for symbol in out_seq:
            if len(symbol) == 1 and symbol.isalpha():
                if symbol in unk_symbol:
                    continue
                else:
                    unk_symbol.append(symbol)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def seg_and_tag(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag(st[:p_start], nums_fraction, nums)
            if nums.count(n) == 1:
                res.append("N" + str(nums.index(n)))
            else:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append("N" + str(nums.index(st_num)))
        else:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        res.append(ss)
    return res


def number_transfer(data):  # transfer num into "NUM"
    '''
    Return:
        processed_datas: list type.
        generate_number: list type, symbols to generate extra.
        copy_nums: int, the count of copied symbol from question to equation.
    '''
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    #pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        equations = d["equation"][2:]

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    nums.append(s[pos.start():pos.end()])
                    input_seq.append("NUM")
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag(equations, nums_fraction, nums)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)

        #copy data
        new_data = d
        new_data["question"] = input_seq
        new_data["equation"] = out_seq
        new_data["number list"] = nums
        new_data["number position"] = num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums
