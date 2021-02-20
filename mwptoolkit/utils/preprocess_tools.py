import re
from copy import deepcopy
from collections import OrderedDict

from mwptoolkit.utils.enum_type import MaskSymbol,NumMask,SpecialTokens

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
        if text_list[i] == '(' and i + 4 < len(text_list) and text_list[
                i + 4] == ')':
            sub = ''.join(text_list[i:i + 5])
            new_list.append(sub)
            i = i + 5
        else:
            new_list.append(text_list[i])
            i += 1
    return new_list

def search_number(seq,equ):
    for idx,symbol in enumerate(equ):
        if symbol in seq:
            continue
        else:
            try:
                number=eval(symbol)
                if number in seq:
                    equ[idx]=str(number)
                else:
                    continue
            except:
                continue
    return equ
def seg_and_tag(st,nums_fraction,nums):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start],nums_fraction,nums)
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:],nums_fraction,nums)
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start],nums_fraction,nums)
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N" + str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:],nums_fraction,nums)
                return res
            for ss in st:
                res.append(ss)
            return res

def seg_and_tag_(st,nums_fraction,nums):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag_(st[:p_start],nums_fraction,nums)
                    try:
                        res.append(nums[n])
                    except:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag_(st[p_end:],nums_fraction,nums)
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag_(st[:p_start],nums_fraction,nums)
                st_num = st[p_start:p_end]
                try:
                    res.append(nums[st_num])
                except:
                    try:
                        number=str(int(eval(st_num)))
                        res.append(nums[number])
                    except:
                        res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag_(st[p_end:],nums_fraction,nums)
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
        nums_fraction = sorted(nums_fraction,
                               key=lambda x: len(x),
                               reverse=True)

        out_seq = seg_and_tag(equations,nums_fraction,nums)
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
        new_data=d
        new_data["question"]=input_seq
        new_data["equation"]=out_seq
        new_data["number list"]=nums
        new_data["number position"]=num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums

def number_transfer_(data,mask_type="NUM",min_generate_keep=0):
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
    if mask_type==MaskSymbol.NUM:
        sent_mask_list=NumMask.NUM
        equ_mask_list=NumMask.number
    elif mask_type==MaskSymbol.alphabet:
        sent_mask_list=NumMask.alphabet
        equ_mask_list=NumMask.alphabet
    elif mask_type==MaskSymbol.number:
        sent_mask_list=NumMask.number
        equ_mask_list=NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx=0
        equ_idx=0
        #nums = []
        nums=OrderedDict()
        num_list=[]
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
                        if mask_type=="NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            nums[s[pos.start():pos.end()]]=equ_mask_list[equ_idx]
                            sent_idx=(sent_idx+1)%len(sent_mask_list)
                            equ_idx=(equ_idx+1)%len(equ_mask_list)
                        else:
                            input_seq.append(nums[s[pos.start():pos.end()]])
                    except:
                        nums[s[pos.start():pos.end()]]=equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx=(equ_idx+1)%len(equ_mask_list)
                        sent_idx=(sent_idx+1)%len(sent_mask_list)
                    finally:
                        num_list.append(s[pos.start():pos.end()])
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count=len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num,mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction,
                               key=lambda x: len(x),
                               reverse=True)

        out_seq = seg_and_tag_(equations,nums_fraction,nums)
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
        new_data=d
        new_data["question"]=input_seq
        new_data["equation"]=out_seq
        new_data["number list"]=num_list
        new_data["number position"]=num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums

def seg_and_tag_mawps(st,nums_fraction,nums):  # seg the equation and tag the num
    res = []
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag_mawps(st[:p_start],nums_fraction,nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag_mawps(st[p_end:],nums_fraction,nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start],nums_fraction,nums)
        st_num = str(eval(st[p_start:p_end]))
        try:
            res.append(nums[st_num])
        except:
            try:
                number=str(int(eval(st_num)))
                if abs(eval(number)-eval(st_num))<1e-4:
                    res.append(nums[number])
                else:
                    res.append(st_num)
            except:
                res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:],nums_fraction,nums)
        return res
    pos_st = re.search("<BRG>",st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag_mawps(st[:p_start],nums_fraction,nums)
        res.append(st[p_start:p_end])
        if p_end < len(st):
            res += seg_and_tag_mawps(st[p_end:],nums_fraction,nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res

def num_transfer_mawps(data,mask_type="number",min_generate_keep=0):
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
    if mask_type==MaskSymbol.NUM:
        sent_mask_list=NumMask.NUM
        equ_mask_list=NumMask.number
    elif mask_type==MaskSymbol.alphabet:
        sent_mask_list=NumMask.alphabet
        equ_mask_list=NumMask.alphabet
    elif mask_type==MaskSymbol.number:
        sent_mask_list=NumMask.number
        equ_mask_list=NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx=0
        equ_idx=0
        #nums = []
        nums=OrderedDict()
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
                    number=str(eval(s[pos.start():pos.end()]))
                    try:
                        if mask_type=="NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number]=equ_mask_list[equ_idx]
                            sent_idx=(sent_idx+1)%len(sent_mask_list)
                            equ_idx=(equ_idx+1)%len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number]=equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx=(equ_idx+1)%len(equ_mask_list)
                        sent_idx=(sent_idx+1)%len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count=len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num,mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction,
                               key=lambda x: len(x),
                               reverse=True)
        # if d["id"]==76:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations,nums_fraction,nums)
        num_list=list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list=list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)

        #copy data
        new_data=d
        new_data["question"]=input_seq
        new_data["equation"]=out_seq
        new_data["number list"]=num_list
        new_data["number position"]=num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums

def num_transfer_multi(data,mask_type="number",min_generate_keep=0,equ_split_symbol=";"):
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
    if mask_type==MaskSymbol.NUM:
        sent_mask_list=NumMask.NUM
        equ_mask_list=NumMask.number
    elif mask_type==MaskSymbol.alphabet:
        sent_mask_list=NumMask.alphabet
        equ_mask_list=NumMask.alphabet
    elif mask_type==MaskSymbol.number:
        sent_mask_list=NumMask.number
        equ_mask_list=NumMask.number

    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    for d in data:
        sent_idx=0
        equ_idx=0
        #nums = []
        nums=OrderedDict()
        #num_list=[]
        input_seq = []
        seg = d["original_text"].split(" ")
        equations = d["equation"]
        equations = re.sub(r"[a-zA-Z]{2,}","x",equations)
        equations = re.sub(equ_split_symbol,SpecialTokens.BRG_TOKEN,equations)

        for s in seg:
            if s == 0:
                input_seq.append(s)
            else:
                pos = re.search(pattern, s)
                if pos and pos.start() == 0:
                    try:
                        number=str(eval(s[pos.start():pos.end()]))
                    except: # "%" in number 
                        number=s[pos.start():pos.end()]
                    try:
                        if mask_type=="NUM":
                            input_seq.append(sent_mask_list[sent_idx])
                            #number=str(eval(s[pos.start():pos.end()]))
                            nums[number]=equ_mask_list[equ_idx]
                            sent_idx=(sent_idx+1)%len(sent_mask_list)
                            equ_idx=(equ_idx+1)%len(equ_mask_list)
                        else:
                            #number=str(eval(s[pos.start():pos.end()]))
                            input_seq.append(nums[number])
                    except:
                        #number=str(eval(s[pos.start():pos.end()]))
                        nums[number]=equ_mask_list[equ_idx]
                        input_seq.append(sent_mask_list[sent_idx])
                        equ_idx=(equ_idx+1)%len(equ_mask_list)
                        sent_idx=(sent_idx+1)%len(sent_mask_list)
                    if pos.end() < len(s):
                        input_seq.append(s[pos.end():])
                else:
                    input_seq.append(s)
        nums_count=len(list(nums.keys()))
        if copy_nums < nums_count:
            copy_nums = nums_count
        nums_fraction = []

        for num,mask in nums.items():
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction,
                               key=lambda x: len(x),
                               reverse=True)
        # if d["id"]==76:
        #     print(1)
        out_seq = seg_and_tag_mawps(equations,nums_fraction,nums)
        num_list=list(nums.keys())
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        mask_num_list=list(nums.values())
        for num in mask_num_list:
            try:
                num_pos.append(input_seq.index(num))
            except:
                continue
        assert len(num_list) == len(num_pos)

        #copy data
        new_data=d
        new_data["question"]=input_seq
        new_data["equation"]=out_seq
        new_data["number list"]=num_list
        new_data["number position"]=num_pos
        processed_datas.append(new_data)

    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums
def from_infix_to_postfix(expression):
    r"""postfix for expression
    """
    st = list()
    res = list()
    priority = {"<BRG>":0,"=":1,"+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
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
    priority = {"<BRG>":0,"=":1,"+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
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
