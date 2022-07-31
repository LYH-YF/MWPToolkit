import re
import copy
from typing import List

from word2number import w2n

from mwptoolkit.utils.enum_type import NumMask, EPT


def trans_symbol_2_number(equ_list, num_list):
    """transfer mask symbol in equation to number.

    Args:
        equ_list (list): equation.
        num_list (list): number list.
    
    Return:
        (list): equation.
    """
    symbol_list = NumMask.number
    new_equ_list = []
    for symbol in equ_list:
        if 'NUM' in symbol:
            index = symbol_list.index(symbol)
            new_equ_list.append(str(num_list[index]))
        else:
            new_equ_list.append(symbol)
    return new_equ_list


def fraction_word_to_num(number_sentence):
    """transfer english expression of fraction to number. numerator and denominator are not more than 10.
    
    Args:
        number_sentence (str): english expression.
    
    Returns:
        (float): number
    """
    fraction = {
        'one-third': 1 / 3, 'one-thirds': 1 / 3, 'one-quarter': 1 / 4, 'one-forth': 1 / 4, 'one-fourth': 1 / 4,
        'one-fourths': 1 / 4, 'one-fifth': 1 / 5, 'one-sixth': 1 / 6, 'one-seventh': 1 / 7, 'one-eighth': 1 / 8,
        'one-ninth': 1 / 9, 'one-tenth': 1 / 10, 'one-fifths': 1 / 5, 'one-sixths': 1 / 6, 'one-sevenths': 1 / 7,
        'one-eighths': 1 / 8, 'one-ninths': 1 / 9, 'one-tenths': 1 / 10, \
        'two-third': 2 / 3, 'two-thirds': 2 / 3, 'two-quarter': 2 / 4, 'two-forth': 2 / 4, 'two-fourth': 2 / 4,
        'two-fourths': 2 / 4, 'two-fifth': 2 / 5, 'two-sixth': 2 / 6, 'two-seventh': 2 / 7, 'two-eighth': 2 / 8,
        'two-ninth': 2 / 9, 'two-tenth': 2 / 10, 'two-fifths': 2 / 5, 'two-sixths': 2 / 6, 'two-sevenths': 2 / 7,
        'two-eighths': 2 / 8, 'two-ninths': 2 / 9, 'two-tenths': 2 / 10, \
        'three-third': 3 / 3, 'three-thirds': 3 / 3, 'three-quarter': 3 / 4, 'three-forth': 3 / 4,
        'three-fourth': 3 / 4, 'three-fourths': 3 / 4, 'three-fifth': 3 / 5, 'three-sixth': 3 / 6,
        'three-seventh': 3 / 7, 'three-eighth': 3 / 8, 'three-ninth': 3 / 9, 'three-tenth': 3 / 10,
        'three-fifths': 3 / 5, 'three-sixths': 3 / 6, 'three-sevenths': 3 / 7, 'three-eighths': 3 / 8,
        'three-ninths': 3 / 9, 'three-tenths': 3 / 10, \
        'four-third': 4 / 3, 'four-thirds': 4 / 3, 'four-quarter': 4 / 4, 'four-forth': 4 / 4, 'four-fourth': 4 / 4,
        'four-fourths': 4 / 4, 'four-fifth': 4 / 5, 'four-sixth': 4 / 6, 'four-seventh': 4 / 7, 'four-eighth': 4 / 8,
        'four-ninth': 4 / 9, 'four-tenth': 4 / 10, 'four-fifths': 4 / 5, 'four-sixths': 4 / 6, 'four-sevenths': 4 / 7,
        'four-eighths': 4 / 8, 'four-ninths': 4 / 9, 'four-tenths': 4 / 10, \
        'five-third': 5 / 3, 'five-thirds': 5 / 3, 'five-quarter': 5 / 4, 'five-forth': 5 / 4, 'five-fourth': 5 / 4,
        'five-fourths': 5 / 4, 'five-fifth': 5 / 5, 'five-sixth': 5 / 6, 'five-seventh': 5 / 7, 'five-eighth': 5 / 8,
        'five-ninth': 5 / 9, 'five-tenth': 5 / 10, 'five-fifths': 5 / 5, 'five-sixths': 5 / 6, 'five-sevenths': 5 / 7,
        'five-eighths': 5 / 8, 'five-ninths': 5 / 9, 'five-tenths': 5 / 10, \
        'six-third': 6 / 3, 'six-thirds': 6 / 3, 'six-quarter': 6 / 4, 'six-forth': 6 / 4, 'six-fourth': 6 / 4,
        'six-fourths': 6 / 4, 'six-fifth': 6 / 5, 'six-sixth': 6 / 6, 'six-seventh': 6 / 7, 'six-eighth': 6 / 8,
        'six-ninth': 6 / 9, 'six-tenth': 6 / 10, 'six-fifths': 6 / 5, 'six-sixths': 6 / 6, 'six-sevenths': 6 / 7,
        'six-eighths': 6 / 8, 'six-ninths': 6 / 9, 'six-tenths': 6 / 10, \
        'seven-third': 7 / 3, 'seven-thirds': 7 / 3, 'seven-quarter': 7 / 4, 'seven-forth': 7 / 4,
        'seven-fourth': 7 / 4, 'seven-fourths': 7 / 4, 'seven-fifth': 7 / 5, 'seven-sixth': 7 / 6,
        'seven-seventh': 7 / 7, 'seven-eighth': 7 / 8, 'seven-ninth': 7 / 9, 'seven-tenth': 7 / 10,
        'seven-fifths': 7 / 5, 'seven-sixths': 7 / 6, 'seven-sevenths': 7 / 7, 'seven-eighths': 7 / 8,
        'seven-ninths': 7 / 9, 'seven-tenths': 7 / 10, \
        'eight-third': 8 / 3, 'eight-thirds': 8 / 3, 'eight-quarter': 8 / 4, 'eight-forth': 8 / 4,
        'eight-fourth': 8 / 4, 'eight-fourths': 8 / 4, 'eight-fifth': 8 / 5, 'eight-sixth': 8 / 6,
        'eight-seventh': 8 / 7, 'eight-eighth': 8 / 8, 'eight-ninth': 8 / 9, 'eight-tenth': 8 / 10,
        'eight-fifths': 8 / 5, 'eight-sixths': 8 / 6, 'eight-sevenths': 8 / 7, 'eight-eighths': 8 / 8,
        'eight-ninths': 8 / 9, 'eight-tenths': 8 / 10, \
        'nine-third': 9 / 3, 'nine-thirds': 9 / 3, 'nine-quarter': 9 / 4, 'nine-forth': 9 / 4, 'nine-fourth': 9 / 4,
        'nine-fourths': 9 / 4, 'nine-fifth': 9 / 5, 'nine-sixth': 9 / 6, 'nine-seventh': 9 / 7, 'nine-eighth': 9 / 8,
        'nine-ninth': 9 / 9, 'nine-tenth': 9 / 10, 'nine-fifths': 9 / 5, 'nine-sixths': 9 / 6, 'nine-sevenths': 9 / 7,
        'nine-eighths': 9 / 8, 'nine-ninths': 9 / 9, 'nine-tenths': 9 / 10
    }
    return fraction[number_sentence.lower()]


def english_word_2_num(sentence_list, fraction_acc=None):
    """transfer english word to number.

    Args:
        sentence_list (list): list of words.
        fraction_acc (int|None): the accuracy to transfer fraction to float, if None, not to match fraction expression.
    
    Returns:
        (list): transfered sentence.
    """
    # bug : 4.9 million can't be matched
    match_word = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
        'billion', 'point'
    ]
    num1 = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    num2 = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    for n2 in num2:
        for n1 in num1:
            match_word.append(n2 + '-' + n1)
    new_list = []
    stack = []
    start_idx = 0
    for idx, word in enumerate(sentence_list):
        if idx < start_idx:
            continue
        if word.lower() in match_word:
            start_idx = idx
            while (sentence_list[start_idx].lower() in match_word):
                stack.append(sentence_list[start_idx])
                start_idx += 1
            if len(stack) == 1 and stack[0] == 'point':
                new_list.append(stack[0])
            elif len(stack) == 1 and stack[0].lower() == 'one':
                new_list.append(stack[0])
            elif len(stack) == 2 and stack[0].lower() == 'one' and stack[1] == 'point':
                new_list.append(stack[0])
                new_list.append(stack[1])
            elif stack[-1] == 'point':
                num_words = ' '.join(stack[:-1])
                number = w2n.word_to_num(num_words)
                new_list.append(str(number))
                new_list.append(stack[-1])
            else:
                if len(stack) >= 2:
                    x = 1
                num_words = ' '.join(stack)
                number = w2n.word_to_num(num_words)
                new_list.append(str(number))
            stack = []
        else:
            new_list.append(word)
    if fraction_acc != None:
        num1 = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        num2 = ['third', 'thirds', 'quarter', 'forth', 'fourth', 'fourths', 'fifth', 'sixth', 'seventh', 'eighth',
                'ninth', 'tenth', 'fifths', 'sixths', 'sevenths', 'eighths', 'ninths', 'tenths']
        match_word = []
        for n1 in num1:
            for n2 in num2:
                match_word.append(n1 + '-' + n2)
        sentence_list = copy.deepcopy(new_list)
        new_list = []
        for idx, word in enumerate(sentence_list):
            if word.lower() in match_word:
                number = fraction_word_to_num(word)
                number = int(number * 10 ** fraction_acc) / 10 ** fraction_acc
                # number=round(number,fraction_acc)
                new_list.append(str(number))
            else:
                new_list.append(word)
    return new_list


def split_number(text_list):
    """separate number expression from other characters.

    Args:
        text_list (list): text list.
    
    Returns:
        (list): processed text list.
    """
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
    """joint fraction number

    Args:
        text_list (list): text list.
    
    Returns:
        (list): processed text list.
    """
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


def joint_number_(text_list):
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


def joint_fraction(text_list: List[str]) -> List[str]:
    """
    joint fraction number

    :param text_list: text list.
    :return: processed text list.
    """
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
    pattern = re.compile("\(\d+/\d+\)")
    new_list_2 = []
    i = 0
    while i < len(new_list):
        if new_list[i].isdigit():
            j = i + 1
            if j < len(new_list) and re.match(pattern, new_list[j]):
                new_list_2.append(new_list[i] + new_list[j])
                i = j + 1
            else:
                new_list_2.append(new_list[i])
                i = i + 1
        else:
            new_list_2.append(new_list[i])
            i += 1
    return new_list_2


def constant_number(const):
    """
    Converts number to constant symbol string (e.g. 'C_3').
    To avoid sympy's automatic simplification of operation over constants.

    :param Union[str,int,float,Expr] const: constant value to be converted.
    :return: (str) Constant symbol string represents given constant.
    """
    if type(const) is str:
        if const in ['C_pi', 'C_e', 'const_pi', 'const_e']:
            # Return pi, e as itself.
            return True, const.replace('const_', 'C_')

        # Otherwise, evaluate string and call this function with the evaluated number
        const = float(const.replace('C_', '').replace('const_', '').replace('_', '.'))
        return constant_number(const)
    elif type(const) is int or int(const) == float(const):
        # If the value is an integer, we trim the following zeros under decimal points.
        return const >= 0, 'C_%s' % int(abs(const))
    else:
        if abs(const - 3.14) < 1E-2:  # Including from 3.14
            return True, 'C_pi'
        if abs(const - 2.7182) < 1E-4:  # Including from 2.7182
            return True, 'C_e'

        # If the value is not an integer, we write it and trim followed zeros.
        # We need to use '%.15f' formatting because str() may gives string using precisions like 1.7E-3
        # Also we will trim after four zeros under the decimal like 0.05000000074 because of float's precision.
        return const >= 0, 'C_%s' % \
               EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', ('%.15f' % abs(const)).replace('.', '_'))
