import re
from copy import deepcopy

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

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
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
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
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
