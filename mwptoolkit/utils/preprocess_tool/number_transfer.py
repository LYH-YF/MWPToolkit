import re
from copy import deepcopy
from collections import OrderedDict
from typing import Tuple

import nltk
from tqdm import tqdm

from mwptoolkit.utils.utils import str2float, lists2dict
from mwptoolkit.utils.enum_type import DatasetName, MaskSymbol, NumMask, SpecialTokens, TaskType
from mwptoolkit.utils.preprocess_tool.number_operator import english_word_2_num, joint_fraction


def number_transfer(datas, dataset_name, task_type, mask_type, min_generate_keep, linear_dataset, equ_split_symbol=';',
                    vocab_level='word', word_lower=False) -> Tuple[list, list, int, list]:
    """
    number transfer

    :param list datas: dataset.
    :param str dataset_name: dataset name.
    :param str task_type: [single_equation | multi_equation], task type.
    :param mask_type:
    :param int min_generate_keep: generate number that count greater than the value, will be kept in output symbols.
    :param bool linear_dataset:
    :param str equ_split_symbol: equation split symbol, in multiple-equation dataset, symbol to split equations, this symbol will be repalced with special token SpecialTokens.BRG
    :param str vocab_level:
    :param bool word_lower:
    :return: processed datas, generate number list, copy number, unk symbol list.
    """
    if dataset_name == DatasetName.math23k:
        transfer = number_transfer_math23k
    elif dataset_name == DatasetName.ape200k:
        transfer = number_transfer_ape200k
    elif dataset_name == DatasetName.asdiv_a:
        transfer = number_transfer_asdiv_a
    elif dataset_name == DatasetName.SVAMP:
        transfer = number_transfer_svamp
    elif dataset_name == DatasetName.mawps_single:
        transfer = number_transfer_mawps_single
    elif dataset_name == DatasetName.mawps:
        transfer = number_transfer_mawps
    elif dataset_name == DatasetName.alg514:
        transfer = num_transfer_alg514
    elif dataset_name == DatasetName.draw:
        transfer = num_transfer_draw
    elif dataset_name == DatasetName.hmwp:
        transfer = num_transfer_hmwp
    else:
        if task_type == TaskType.SingleEquation:
            transfer = number_transfer_single
        elif task_type == TaskType.MultiEquation:
            transfer = num_transfer_multi
        else:
            raise NotImplementedError
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    unk_symbol = []
    for data in tqdm(datas,desc='word segmentation and number mapping'):
        if task_type == TaskType.SingleEquation:
            new_data = transfer(data, mask_type, linear_dataset, vocab_level, word_lower)
        elif task_type == TaskType.MultiEquation:
            new_data = transfer(data, mask_type, equ_split_symbol, vocab_level, word_lower)
        else:
            raise NotImplementedError
        if dataset_name == DatasetName.mawps_single and task_type == TaskType.SingleEquation and '=' in new_data[
            "equation"]:
            continue
        num_list = new_data["number list"]
        out_seq = new_data["equation"]
        copy_num = len(new_data["number list"])

        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0] == '-' and len(s) >= 2 and s[1].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        if copy_num > copy_nums:
            copy_nums = copy_num

        # get unknown number
        if task_type == TaskType.SingleEquation:
            if linear_dataset:
                for s in out_seq:
                    if len(s) == 1 and s.isalpha():
                        if s in unk_symbol:
                            continue
                        else:
                            unk_symbol.append(s)
            else:
                pass
        elif task_type == TaskType.MultiEquation:
            for s in out_seq:
                if len(s) == 1 and s.isalpha():
                    if s in unk_symbol:
                        continue
                    else:
                        unk_symbol.append(s)
        else:
            raise NotImplementedError

        processed_datas.append(new_data)
    # keep generate number
    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol


def seg_and_tag_single(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"-\d+\.\d+%?|-\d+%?", st)  # search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_single(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_single(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_single(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_single(st[p_end:], nums_fraction, nums)
            return res

    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_single(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_single(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss == ' ':
            continue
        res.append(ss)
    return res


def seg_and_tag_math23k(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-(([(]\d+\.\d+[)])|([(]\d+/\d+[)]))",
                       st)  # search negative number but filtate minus symbol
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
                number = str(str2float(st_num))
                try:
                    res.append(nums[number])
                except:
                    res.append(number)
        if p_end < len(st):
            res += seg_and_tag_asdiv_a(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss == ' ':
            continue
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
            number = str(str2float((st_num)))
            try:
                res.append(nums[number])
            except:
                res.append(number)
        if p_end < len(st):
            res += seg_and_tag_svamp(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss == " ":
            continue
        res.append(ss)
    return res


def seg_and_tag_multi(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  # search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  # search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_multi(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)\s-\s((\d+\.?\d*))", st)  # search negative number but filtate minus symbol
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
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  # search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_hmwp(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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


def seg_and_tag_mawps_single(st, nums_fraction, nums):
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  # search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps_single(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
        if p_end < len(st):
            res += seg_and_tag_mawps_single(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_mawps_single(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_mawps_single(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  # search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps_single(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
        if p_end < len(st):
            res += seg_and_tag_mawps_single(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res


def seg_and_tag_mawps(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  # search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  # search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
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


def number_transfer_single(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    if word_lower:
        data["question"] = data["question"].lower()
    seg = data["question"].split(" ")
    equations = data["equation"]
    if linear:
        if equations.startswith('x=') or equations.startswith('X='):
            equations = equations[2:]
        elif equations.endswith('=x') or equations.endswith('=X'):
            equations = equations[:-2]

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(s[pos.start():pos.end()])
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '　' or s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_single(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        source[pos] = num_str
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos

    return new_data


def number_transfer_math23k(data, mask_type, linear, vocab_level='word', word_lower=False):
    # pattern = re.compile("\data*\(\data+/\data+\)\data*|\data+\.\data+%?|\data+%?")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    if word_lower:
        data["segmented_text"] = data["segmented_text"].lower()
    seg = data["segmented_text"].split(" ")
    equations = data["equation"][2:]
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
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '　' or s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_math23k(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos

    return new_data


def number_transfer_ape200k(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    if word_lower:
        data["segmented_text"] = data["segmented_text"].lower()
    seg = data["segmented_text"].split(" ")
    seg = joint_fraction(seg)
    equations = data["equation"]
    if "x=" == equations[:2] or "X=" == equations[:2]:
        equations = equations[2:]
    equations = equations.replace('**','^',100)
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(s[pos.start():pos.end()])
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '　' or s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)
    out_seq_ = seg_and_tag_ape200k(equations, nums_fraction, nums)
    out_seq = []
    i = 0
    while i<len(out_seq_):
        s = out_seq_[i]
        if s == '%':
            out_seq.append('/')
            out_seq.append('100')
            i+=1
        elif s == ':':
            out_seq.append('/')
            i+=1
        else:
            out_seq.append(s)
            i+=1

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos

    return new_data


def number_transfer_asdiv_a(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    if word_lower:
        data["Body"] = data["Body"].lower()
        data["Question"] = data["Question"].lower()
    seg = nltk.word_tokenize(data["Body"] + ' ' + data["Question"])
    formula = data["Formula"]
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
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '　' or s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_asdiv_a(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    new_data = data
    new_data['id'] = data['@ID']
    new_data['ans'] = ans
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos

    return new_data


def number_transfer_svamp(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    if word_lower:
        data["Body"] = data["Body"].lower()
        data["Question"] = data["Question"].lower()
    seg = nltk.word_tokenize(data["Body"] + ' ' + data["Question"])
    equations = data["Equation"]
    if equations.startswith('( ') and equations.endswith(' )'):
        equations = equations[2:-2]

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_svamp(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    new_data["id"] = data["ID"]
    new_data["ans"] = data["Answer"]

    return new_data


def number_transfer_mawps_single(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    if word_lower:
        data["sQuestion"] = data["sQuestion"].lower()
    seg = nltk.word_tokenize(data["sQuestion"])
    equations = data["lEquations"][0]
    if equations[:2] == 'x=' or equations[:2] == 'X=':
        equations = equations[2:]
    if equations[-2:] == '=x' or equations[-2:] == '=X':
        equations = equations[:-2]

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '　' or s == '':
                continue
            input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_mawps_single(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    new_data = data
    new_data['id'] = data['iIndex']
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    new_data["ans"] = data['lSolutions'][0]

    return new_data


def number_transfer_mawps(data, mask_type, linear, vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    if word_lower:
        data["original_text"] = data["original_text"].lower()
    seg = data["original_text"].split(" ")
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)
    if data['id'] == 46:
        x = 1
    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_mawps(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    # copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    return new_data


def num_transfer_multi(data, mask_type, equ_split_symbol=";", vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    if word_lower:
        data["original_text"] = data["original_text"].lower()
    seg = data["original_text"].split(" ")
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
    equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    # copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    return new_data


def num_transfer_alg514(data, mask_type, equ_split_symbol=";", vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    if word_lower:
        data["original_text"] = data["original_text"].lower()
    seg = nltk.word_tokenize(data["original_text"])
    for idx, word in enumerate(seg):
        if re.match(r"(\d+\,\d+)+", word):
            new_word = "".join(word.split(","))
            seg[idx] = new_word
    seg = english_word_2_num(seg)
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
    equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            # input_seq.append(s[pos.start():pos.end()])
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)
    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    # copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    if num_list == []:
        new_data["number list"] = ["-inf"]
        new_data["number position"] = [-1]

    return new_data


def num_transfer_draw(data, mask_type, equ_split_symbol=";", vocab_level='word', word_lower=False):
    # pattern = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")
    pattern = re.compile(r"\d+\/\d+|\d+\.\d+%?|\d+%?|(-\d+)")

    if word_lower:
        data["original_text"] = data["original_text"].lower()
    seg = data["original_text"].split(" ")
    for idx, word in enumerate(seg):
        if re.match(r"(\d+\,\d+)+", word):
            new_word = "".join(word.split(","))
            seg[idx] = new_word
        elif re.match(r"\.\d+", word):
            new_word = "0" + word
            seg[idx] = new_word
    seg = english_word_2_num(seg, 3)
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
    equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)
    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = []
    pos_st = re.search(r"^-((\d+\.?\d*))", equations)  # search negative number starting
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            out_seq += seg_and_tag_multi(equations[:p_start], nums_fraction, nums)
        st_num = equations[p_start:p_end]
        try:
            out_seq.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    out_seq.append(nums[number])
                else:
                    out_seq.append(number)
            except:
                out_seq.append(number)
        if p_end < len(equations):
            out_seq += seg_and_tag_multi(equations[p_end:], nums_fraction, nums)
    else:
        out_seq = seg_and_tag_multi(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    # copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    if num_list == []:
        new_data["number list"] = ["-inf"]
        new_data["number position"] = [-1]

    return new_data


def num_transfer_hmwp(data, mask_type, equ_split_symbol=";", vocab_level='word', word_lower=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    if word_lower:
        data["original_text"] = data["original_text"].lower()
    seg = data["original_text"].split(" ")
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)
    equations = re.sub(equ_split_symbol, SpecialTokens.BRG_TOKEN, equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            # input_seq.append(s[pos.start():pos.end()])
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq,
                                                                                                          mask_type,
                                                                                                          pattern)

    out_seq = seg_and_tag_hmwp(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)
    # copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos

    return new_data


def get_num_pos(input_seq, mask_type, pattern):
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number
    nums = OrderedDict()
    num_list = []
    num_pos = []
    num_pos_dict = {}

    if mask_type == MaskSymbol.NUM:
        # find all number position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                num_list.append(word)
                num_pos.append(word_pos)
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_pos_dict[word] = [word_pos]

        mask_list = equ_mask_list[:len(num_list)]
        new_num_list = []
        new_mask_list = []
        for i in num_list:
            if num_list.count(i) != 1:
                x = 1
            if num_list.count(i) == 1:
                new_num_list.append(i)
                new_mask_list.append(mask_list[num_list.index(i)])
            else:
                pass
        nums = lists2dict(new_num_list, new_mask_list)
    else:
        # find all number position
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

    # all number position
    all_pos = []
    if mask_type == MaskSymbol.NUM:
        all_pos = deepcopy(num_pos)
    else:
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                all_pos.append(pos)

    # final numbor position
    final_pos = []
    if mask_type == MaskSymbol.NUM:
        final_pos = deepcopy(num_pos)
    else:
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            final_pos.append(max(num_pos_dict[num]))

    # number transform
    for num, mask in nums_for_ques.items():
        for pos in num_pos_dict[num]:
            input_seq[pos] = mask

    # nums_fraction = []
    # for num, mask in nums.items():
    #     if re.search("\data*\(\data+/\data+\)\data*", num):
    #         nums_fraction.append(num)
    # nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
    nums_fraction = []
    for num, mask in nums.items():
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    return input_seq, num_list, final_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction
