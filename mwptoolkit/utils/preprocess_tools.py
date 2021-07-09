import re
import json
import random
from copy import deepcopy
from collections import OrderedDict

import nltk
import stanza
from word2number import w2n

from mwptoolkit.utils.utils import read_json_data, str2float, lists2dict
from mwptoolkit.utils.enum_type import MaskSymbol, NumMask, SpecialTokens
from mwptoolkit.utils.data_structure import DependencyTree


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


def joint_number_(text_list):  # match longer fraction such as ( 1 / 1000000 )
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


def seg_and_tag_svamp(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_svamp(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_svamp(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_svamp(st[:p_start], nums_fraction, nums)
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
            res += seg_and_tag_svamp(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss == " ":
            continue
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


def seg_and_tag_asdiv_a(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_asdiv_a(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_asdiv_a(st[p_end:], nums_fraction, nums)
            return res

    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_asdiv_a(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(eval(st_num)))
                res.append(nums[number])
            except:
                try:
                    number = str(str2float(st_num))
                    res.append(nums[number])
                except:
                    res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_asdiv_a(st[p_end:], nums_fraction, nums)
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
            res += seg_and_tag_math23k(st[:p_start], nums_fraction, nums)
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
            res += seg_and_tag_math23k(st[p_end:], nums_fraction, nums)
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
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  #search negative number but filtate minus symbol
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


def seg_and_tag_multi(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  #search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(str2float(st_num))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_multi(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_multi(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  #search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(str2float(st_num))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_multi(st[p_end:], nums_fraction, nums)
        return res
    pos_st = re.search("<BRG>", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
        res.append(st[p_start:p_end])
        if p_end < len(st):
            res += seg_and_tag_multi(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res


def seg_and_tag_hmwp(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)\s-\s((\d+\.?\d*))", st)  #search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 2
        p_end = pos_st.end()
        num_str = ''.join(st[p_start:p_end].split(" "))
        if p_start > 0:
            res += seg_and_tag_hmwp(st[:p_start], nums_fraction, nums)
        st_num = num_str
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(str2float(st_num)))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_hmwp(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_hmwp(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_hmwp(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  #search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_hmwp(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            try:
                number = str(int(str2float(st_num)))
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_hmwp(st[p_end:], nums_fraction, nums)
        return res
    pos_st = re.search("<BRG>", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_hmwp(st[:p_start], nums_fraction, nums)
        res.append(st[p_start:p_end])
        if p_end < len(st):
            res += seg_and_tag_hmwp(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res


def number_transfer_math23k(data, mask_type="number", min_generate_keep=0):
    r'''transfer num process

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
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        equations = d["equation"][2:]
        if '千' in equations:
            equations = equations[:equations.index('千')]
        num_pos_dict = {}
        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(s[pos.start():pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if s == '　' or s == '':
                    continue
                input_seq.append(s)
        # find all num position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        #input_seq = deepcopy(seg)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_math23k(equations, nums_fraction, nums)
        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

            if mask_type == MaskSymbol.NUM:
                if 'NUM' in s:
                    number = num_list[int(s[4:])]
                    if len(num_pos_dict[number]) > 1:
                        out_seq[idx] = number

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        #get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
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


def number_transfer_ape200k(data, mask_type="number", min_generate_keep=0):
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
        num_pos_dict = {}
        num_list = []
        input_seq = []
        seg = d["segmented_text"].split(" ")
        seg = joint_number_(seg)
        equations = d["equation"]
        if "x=" == equations[:2] or "X=" == equations[:2]:
            equations = equations[2:]
        # if '千' in equations:
        #     equations = equations[:equations.index('千')]

        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(s[pos.start():pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # find all num position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # equation tag
        out_seq = seg_and_tag_ape200k(equations, nums_fraction, nums)

        # tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)

        #get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)

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


def number_transfer_svamp(data, mask_type="number", min_generate_keep=0):
    r'''transfer num process

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
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["Body"].split(" ") + d["Question"].split()
        equations = d["Equation"]
        if equations.startswith('( ') and equations.endswith(' )'):
            equations = equations[2:-2]

        num_pos_dict = {}
        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(s[pos.start():pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # find all num position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        #input_seq = deepcopy(seg)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_svamp(equations, nums_fraction, nums)
        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

            if mask_type == MaskSymbol.NUM:
                if 'NUM' in s:
                    number = num_list[int(s[4:])]
                    if len(num_pos_dict[number]) > 1:
                        out_seq[idx] = number

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        #get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
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
        new_data["id"] = d["ID"]
        new_data["ans"] = d["Answer"]
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums


def number_transfer_asdiv_a(data, mask_type="number", min_generate_keep=0):
    r'''transfer num process

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
        #nums = []
        nums = OrderedDict()
        num_list = []
        input_seq = []
        seg = d["Body"].split(" ") + d["Question"].split()
        #sss=d["Body"]+d["Question"]
        seg = nltk.word_tokenize(d["Body"]+' '+d["Question"])
        formula = d["Formula"]
        equations = formula[:formula.index('=')]
        ans = formula[formula.index('=') + 1:]
        num_pos_dict = {}
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
        seg = english_word_2_num(seg)
        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(s[pos.start():pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            elif pos and pos.start() > 0:
                input_seq.append(s[:pos.start()])
                input_seq.append(s[pos.start():pos.end()])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # find all num position
        for word_pos, word in enumerate(input_seq):
            if word[0].isdigit() and '.' in word:
                word = str(str2float(word))
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        #input_seq = deepcopy(seg)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_asdiv_a(equations, nums_fraction, nums)
        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

            if mask_type == MaskSymbol.NUM:
                if 'NUM' in s:
                    number = num_list[int(s[4:])]
                    if len(num_pos_dict[number]) > 1:
                        out_seq[idx] = number

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        #get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)
        #copy data
        # if d["id"]=="8883":
        #     print(1)
        new_data = d
        new_data['id'] = d['@ID']
        new_data['ans'] = ans
        new_data["question"] = input_seq
        new_data["ques source 1"] = source
        new_data["equation"] = out_seq
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
    return processed_datas, generate_number, copy_nums


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
        # if d["id"]==5580:
        #     print(1)
        sent_idx = 0
        equ_idx = 0
        #nums = []
        nums = OrderedDict()
        num_pos_dict = {}
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        for idx, word in enumerate(seg):
            if re.match(r"(\d+\,\d+)+", word):
                new_word = "".join(word.split(","))
                seg[idx] = new_word
        seg = english_word_2_num(seg)
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                #input_seq.append(s[pos.start():pos.end()])
                input_seq.append(str(str2float(s[pos.start():pos.end()])))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # find all num position
        num_list = []
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        #input_seq = deepcopy(seg)
        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

        # tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and str(str2float(s)) not in generate_nums and s not in num_list:
                s = str(str2float(s))
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        # get unknown number
        for s in out_seq:
            if len(s) == 1 and s.isalpha():
                if s in unk_symbol:
                    continue
                else:
                    unk_symbol.append(s)

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        # get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)

        # copy data
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
        sent_idx = 0
        equ_idx = 0
        nums = OrderedDict()
        num_pos_dict = {}
        num_list = []
        input_seq = []
        seg = d["original_text"].split(" ")
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(str(str2float(s[pos.start():pos.end()])))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if s == '':
                    continue
                input_seq.append(s)

        # find all num position
        num_list = []
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

        # tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        # get unknown number
        for s in out_seq:
            if len(s) == 1 and s.isalpha():
                if s in unk_symbol:
                    continue
                else:
                    unk_symbol.append(s)

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        # get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)

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

    pattern = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")
    pattern = re.compile(r"\d+\/\d+|\d+\.\d+%?|\d+%?|(-\d+)")

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
        num_pos_dict = {}
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
        seg = english_word_2_num(seg)
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                input_seq.append(str(str2float(s[pos.start():pos.end()])))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # find all num position
        num_list = []
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = []
        pos_st = re.search(r"^-((\d+\.?\d*))", equations)  #search negative number starting
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                out_seq += seg_and_tag_multi(equations[:p_start], nums_fraction, nums)
            st_num = equations[p_start:p_end]
            try:
                out_seq.append(nums[st_num])
            except:
                try:
                    number = str(int(str2float(st_num)))
                    if abs(eval(number) - eval(st_num)) < 1e-4:
                        out_seq.append(nums[number])
                    else:
                        out_seq.append(st_num)
                except:
                    out_seq.append(st_num)
            if p_end < len(equations):
                out_seq += seg_and_tag_multi(equations[p_end:], nums_fraction, nums)
        else:
            out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

        # tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and str(str2float(s)) not in generate_nums and s not in num_list:
                s = str(str2float(s))
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if re.match(r"^-((\d+\.?\d*))", s) and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        # get unknown number
        for s in out_seq:
            if len(s) == 1 and s.isalpha():
                if s in unk_symbol:
                    continue
                else:
                    unk_symbol.append(s)

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        # get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)

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
        num_pos_dict = {}
        input_seq = []
        seg = d["original_text"].split(" ")
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
        equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

        # match and split number
        input_seq = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                #input_seq.append(s[pos.start():pos.end()])
                input_seq.append(str(str2float(s[pos.start():pos.end()])))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        # if d['id']==15538078:
        #     print(1)
        # find all num position
        num_list = []
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])
        nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

        all_pos = []
        # number transform
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
                all_pos.append(pos)

        nums_count = len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count

        nums_fraction = []
        for num, mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        out_seq = seg_and_tag_hmwp(equations, nums_fraction, nums)
        # tag the num which is generated
        for s in out_seq:
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        # get unknown number
        for s in out_seq:
            if len(s) == 1 and s.isalpha():
                if s in unk_symbol:
                    continue
                else:
                    unk_symbol.append(s)

        source = deepcopy(input_seq)
        for pos in all_pos:
            for key, value in num_pos_dict.items():
                if pos in value:
                    num_str = key
                    break
            num = str(str2float(num_str))
            source[pos] = num
        source = ' '.join(source)
        # get final number position
        num_pos = []
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            num_pos.append(max(num_pos_dict[num]))
        assert len(num_list) == len(num_pos)
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


def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


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
        if (i + 4) < len(equ_list) and 'NUM' in equ_list[i] or equ_list[i].isalpha() and '+' in equ_list[i + 1] and 'NUM' in equ_list[i + 2] and '+' in equ_list[i + 3] and 'NUM' in equ_list[i + 4]:
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
        elif (i + 4) < len(equ_list) and 'NUM' in equ_list[i] and '*' in equ_list[i + 1] and 'NUM' in equ_list[i + 2] and '*' in equ_list[i + 3] and 'NUM' in equ_list[i + 4]:
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
        elif (i + 2) < len(equ_list) and 'NUM' in equ_list[i] and '+' in equ_list[i + 1] and 'NUM' in equ_list[i + 2]:
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
        elif (i + 2) < len(equ_list) and 'NUM' in equ_list[i] and '*' in equ_list[i + 1] and 'NUM' in equ_list[i + 2]:
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


def EN_rule2_(equ_list):
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


def deprel_tree_to_file(train_datas, valid_datas, test_datas, path, language, use_gpu):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error', use_gpu=use_gpu)
    new_datas = []
    for idx, data in enumerate(train_datas):
        doc = nlp(data["ques source 1"])
        token_list = doc.to_dict()[0]
        new_datas.append({'id': data['id'], 'deprel': token_list})
    for idx, data in enumerate(valid_datas):
        doc = nlp(data["ques source 1"])
        token_list = doc.to_dict()[0]
        new_datas.append({'id': data['id'], 'deprel': token_list})
    for idx, data in enumerate(test_datas):
        doc = nlp(data["ques source 1"])
        token_list = doc.to_dict()[0]
        new_datas.append({'id': data['id'], 'deprel': token_list})
    write_json_data(new_datas, path)


def get_group_nums_(train_datas, valid_datas, test_datas, path):
    deprel_datas = read_json_data(path)
    new_datas = []
    for idx, data in enumerate(train_datas):
        group_nums = []
        num_pos = data["number position"]
        sent_len = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
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
                        upos = token_list[pos]['upos']
                        head_pos = token_list[pos]['head'] - 1
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
                deprel_datas.remove(deprel_data)
                break
        data["group nums"] = group_nums
    for idx, data in enumerate(valid_datas):
        group_nums = []
        num_pos = data["number position"]
        sent_len = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
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
                        upos = token_list[pos]['upos']
                        head_pos = token_list[pos]['head'] - 1
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
                deprel_datas.remove(deprel_data)
                break
        data["group nums"] = group_nums
    for idx, data in enumerate(test_datas):
        group_nums = []
        num_pos = data["number position"]
        sent_len = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
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
                        upos = token_list[pos]['upos']
                        head_pos = token_list[pos]['head'] - 1
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
                deprel_datas.remove(deprel_data)
                break
        data["group nums"] = group_nums

    return train_datas, valid_datas, test_datas


def span_level_deprel_tree_to_file(train_datas, valid_datas, test_datas, path, language, use_gpu):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
    new_datas = []
    max_span_size = 0
    for idx, data in enumerate(train_datas):
        sentences = split_sentence(data["ques source 1"])
        dependency_infos = []
        deprel_trees = []
        for sentence in sentences:
            dependency_info = []
            doc = nlp(sentence)
            token_list = doc.to_dict()[0]
            for token in token_list:
                deprel = token['deprel']
                father_idx = token['head'] - 1
                child_idx = token['id'] - 1
                dependency_info.append([deprel, child_idx, father_idx])
            dependency_infos.append(dependency_info)
        new_datas.append({'id': data['id'], 'split sentences source': sentences, 'dependency info': dependency_infos})
    for idx, data in enumerate(valid_datas):
        sentences = split_sentence(data["ques source 1"])
        dependency_infos = []
        deprel_trees = []
        for sentence in sentences:
            dependency_info = []
            doc = nlp(sentence)
            token_list = doc.to_dict()[0]
            for token in token_list:
                deprel = token['deprel']
                father_idx = token['head'] - 1
                child_idx = token['id'] - 1
                dependency_info.append([deprel, child_idx, father_idx])
            dependency_infos.append(dependency_info)
        new_datas.append({'id': data['id'], 'split sentences source': sentences, 'dependency info': dependency_infos})
    for idx, data in enumerate(test_datas):
        sentences = split_sentence(data["ques source 1"])
        dependency_infos = []
        deprel_trees = []
        for sentence in sentences:
            dependency_info = []
            doc = nlp(sentence)
            token_list = doc.to_dict()[0]
            for token in token_list:
                deprel = token['deprel']
                father_idx = token['head'] - 1
                child_idx = token['id'] - 1
                dependency_info.append([deprel, child_idx, father_idx])
            dependency_infos.append(dependency_info)
        new_datas.append({'id': data['id'], 'split sentences source': sentences, 'dependency info': dependency_infos})
    write_json_data(new_datas, path)


def get_span_level_deprel_tree_(train_datas, valid_datas, test_datas, path):
    deprel_datas = read_json_data(path)
    max_span_size = 0
    for idx, data in enumerate(train_datas):
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                masked_sentences = split_sentence(' '.join(data['question']))
                span_size = len(masked_sentences)
                if span_size > max_span_size:
                    max_span_size = span_size
                deprel_trees = []
                for sentence, dependency_info in zip(deprel_data['split sentences source'], deprel_data['dependency info']):
                    tree = DependencyTree()
                    tree.sentence2tree(sentence.split(' '), dependency_info)
                    deprel_trees.append(tree)
                data['split sentences'] = [sentence.split(' ') for sentence in masked_sentences]
                data['split sentences source'] = deprel_data['split sentences source']
                data['dependency info'] = deprel_data['dependency info']
                data['deprel tree'] = deprel_trees
                deprel_datas.remove(deprel_data)
                break
    for idx, data in enumerate(valid_datas):
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                masked_sentences = split_sentence(' '.join(data['question']))
                span_size = len(masked_sentences)
                if span_size > max_span_size:
                    max_span_size = span_size
                deprel_trees = []
                for sentence, dependency_info in zip(deprel_data['split sentences source'], deprel_data['dependency info']):
                    tree = DependencyTree()
                    tree.sentence2tree(sentence.split(' '), dependency_info)
                    deprel_trees.append(tree)
                data['split sentences'] = [sentence.split(' ') for sentence in masked_sentences]
                data['split sentences source'] = deprel_data['split sentences source']
                data['dependency info'] = deprel_data['dependency info']
                data['deprel tree'] = deprel_trees
                deprel_datas.remove(deprel_data)
                break
    for idx, data in enumerate(test_datas):
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                masked_sentences = split_sentence(' '.join(data['question']))
                span_size = len(masked_sentences)
                if span_size > max_span_size:
                    max_span_size = span_size
                deprel_trees = []
                for sentence, dependency_info in zip(deprel_data['split sentences source'], deprel_data['dependency info']):
                    tree = DependencyTree()
                    tree.sentence2tree(sentence.split(' '), dependency_info)
                    deprel_trees.append(tree)
                data['split sentences'] = [sentence.split(' ') for sentence in masked_sentences]
                data['split sentences source'] = deprel_data['split sentences source']
                data['dependency info'] = deprel_data['dependency info']
                data['deprel tree'] = deprel_trees
                deprel_datas.remove(deprel_data)
                break
    return train_datas, valid_datas, test_datas, max_span_size

    #token_list=deprel_data['deprel']


def get_deprel_tree_(train_datas, valid_datas, test_datas, path):
    deprel_datas = read_json_data(path)
    deprel_tokens = []
    for idx, data in enumerate(train_datas):
        group_nums = []
        deprel_token = []
        length = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
                for idx, x in enumerate(token_list):
                    token = x['deprel']
                    if token in deprel_token:
                        deprel_idx = deprel_token.index(token) + length
                    else:
                        deprel_token.append(token)
                        deprel_idx = deprel_token.index(token) + length
                    group_nums.append([x['head'] - 1, deprel_idx])
                    group_nums.append([deprel_idx, idx])
                data["group nums"] = group_nums
                data["question"] = data["question"] + deprel_token
                for token in deprel_token:
                    if token not in deprel_tokens:
                        deprel_tokens.append(token)
                deprel_datas.remove(deprel_data)
                break
    for idx, data in enumerate(valid_datas):
        group_nums = []
        deprel_token = []
        length = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
                for idx, x in enumerate(token_list):
                    token = x['deprel']
                    if token in deprel_token:
                        deprel_idx = deprel_token.index(token) + length
                    else:
                        deprel_token.append(token)
                        deprel_idx = deprel_token.index(token) + length
                    group_nums.append([x['head'] - 1, deprel_idx])
                    group_nums.append([deprel_idx, idx])
                data["group nums"] = group_nums
                data["question"] = data["question"] + deprel_token
                deprel_datas.remove(deprel_data)
                break
    for idx, data in enumerate(test_datas):
        group_nums = []
        deprel_token = []
        length = len(data["question"])
        for deprel_data in deprel_datas:
            if data['id'] != deprel_data['id']:
                continue
            else:
                token_list = deprel_data['deprel']
                for idx, x in enumerate(token_list):
                    token = x['deprel']
                    if token in deprel_token:
                        deprel_idx = deprel_token.index(token) + length
                    else:
                        deprel_token.append(token)
                        deprel_idx = deprel_token.index(token) + length
                    group_nums.append([x['head'] - 1, deprel_idx])
                    group_nums.append([deprel_idx, idx])
                data["group nums"] = group_nums
                data["question"] = data["question"] + deprel_token
                deprel_datas.remove(deprel_data)
                break
    return train_datas, valid_datas, test_datas, deprel_tokens


def get_group_nums(datas, language, use_gpu):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error', use_gpu=use_gpu)
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
                upos = token_list[pos]['upos']
                head_pos = token_list[pos]['head'] - 1
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
        # group_words=[]
        # for group_num in group_nums:
        #     group_word=[]
        #     for idx in group_num:
        #         group_word.append(token_list[idx]["text"])
        #     group_words.append(group_word)
        # path="/group_nums.json"
        # json_data={"sentence":data["ques source 1"],"num pos":num_pos,"words":group_words}
        # write_json_data(json_data,path)
    return new_datas


def get_deprel_tree(datas, language):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
    new_datas = []
    deprel_tokens = []
    for idx, data in enumerate(datas):
        group_nums = []
        deprel_token = []
        doc = nlp(data["ques source 1"])
        token_list = doc.to_dict()[0]
        length = len(data["question"])
        for idx, x in enumerate(token_list):
            token = x['deprel']
            if token in deprel_token:
                deprel_idx = deprel_token.index(token) + length
            else:
                deprel_token.append(token)
                deprel_idx = deprel_token.index(token) + length
            group_nums.append([x['head'] - 1, deprel_idx])
            group_nums.append([deprel_idx, idx])
        data["group nums"] = group_nums
        data["question"] = data["question"] + deprel_token
        new_datas.append(data)
        for token in deprel_token:
            if token not in deprel_tokens:
                deprel_tokens.append(token)
    return new_datas, deprel_tokens


def get_span_level_deprel_tree(datas, language):
    nlp = stanza.Pipeline(language, processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error')
    new_datas = []
    max_span_size = 0
    for idx, data in enumerate(datas):
        sentences = split_sentence(data["ques source 1"])
        masked_sentences = split_sentence(' '.join(data['question']))
        span_size = len(masked_sentences)
        if span_size > max_span_size:
            max_span_size = span_size
        dependency_infos = []
        deprel_trees = []
        for sentence in sentences:
            dependency_info = []
            doc = nlp(sentence)
            token_list = doc.to_dict()[0]
            for token in token_list:
                deprel = token['deprel']
                father_idx = token['head'] - 1
                child_idx = token['id'] - 1
                dependency_info.append([deprel, child_idx, father_idx])
            tree = DependencyTree()
            tree.sentence2tree(sentence.split(' '), dependency_info)
            dependency_infos.append(dependency_info)
            deprel_trees.append(tree)
        data['split sentences'] = [sentence.split(' ') for sentence in masked_sentences]
        data['split sentences source'] = sentences
        data['deprel tree'] = deprel_trees
        data['dependency info'] = dependency_infos
        new_datas.append(data)
    return new_datas, max_span_size


def split_sentence(text):
    #seps = ['，',',','。','．','. ','；','？','！','!']
    sentences = nltk.tokenize.sent_tokenize(text)
    #seps='，。(\. )．；？！!'
    #x=f"([{seps}])"
    #seps = "，。．.；？！!"
    spans_posts = []
    seps = "，。．；？！!"
    sep_pattern = re.compile(f"([{seps}])")
    #sep_pattern = re.compile(r'，|。|(\. )|．|；|？|！|!',re.S)
    for sentence in sentences:
        spans = re.split(sep_pattern, sentence)
        spans = [span.strip() for span in spans if span.strip() != '']
        spans_post = []
        for i, span in enumerate(spans):
            if span in seps:
                if i > 0 and spans[i - 1] not in seps:
                    spans_post[-1] += ' ' + span
            else:
                spans_post.append(span)
        spans_posts += spans_post
    return spans_posts


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


def id_reedit(trainset, validset, testset):
    r"""if some datas of a dataset hava the same id, re-edit the id for differentiate them. 

    example: There are two datas have the same id 709356. Make one of them be 709356 and the other be 709356-1.
    """
    id_dict = {}
    for data in trainset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in validset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in testset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in trainset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    for data in validset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    for data in testset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    return trainset, validset, testset


def english_word_2_num(sentence_list):
    # bug : 4.9 million can't be matched
    match_word=[
        'zero','one','two','three','four','five','six','seven','eight','nine','ten',\
        'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen',\
        'twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety',\
        'hundred','thousand','million','billion',\
        'point'
    ]
    num1=['one','two','three','four','five','six','seven','eight','nine']
    num2=['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
    for n2 in num2:
        for n1 in num1:
            match_word.append(n2+'-'+n1)
    new_list=[]
    stack=[]
    start_idx=0
    for idx,word in enumerate(sentence_list):
        if idx<start_idx:
            continue
        if word.lower() in match_word :
            start_idx=idx
            while(sentence_list[start_idx].lower() in match_word):
                stack.append(sentence_list[start_idx])
                start_idx+=1
            if len(stack)==1 and stack[0] == 'point':
                new_list.append(stack[0])
            # elif len(stack)==2 and 'point' in stack and 'and' in stack:
            #     new_list.extend(stack)
            elif stack[-1] == 'point':
                num_words=' '.join(stack[:-1])
                number=w2n.word_to_num(num_words)
                new_list.append(str(number))
                new_list.append(stack[-1])
            else:
                if len(stack)>=2:
                    x=1
                num_words=' '.join(stack)
                number=w2n.word_to_num(num_words)
                new_list.append(str(number))
            stack=[]
        else:
            new_list.append(word)
    return new_list


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
