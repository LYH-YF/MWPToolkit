import copy
from word2number import w2n

from mwptoolkit.utils.enum_type import NumMask

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
def word_to_num(number_sentence):
    ['one-third', 'one-quarter', 'one-forth', 'one-fifth', 'one-sixth', 'one-seventh', 'one-eighth', 'one-ninth', 'one-tenth', 'two-third', 'two-quarter', 'two-forth', 'two-fifth', 'two-sixth', 'two-seventh', 'two-eighth', 'two-ninth', 'two-tenth', 'three-third', 'three-quarter', 'three-forth', 'three-fifth', 'three-sixth', 'three-seventh', 'three-eighth', 'three-ninth', 'three-tenth', 'four-third', 'four-quarter', 'four-forth', 'four-fifth', 'four-sixth', 'four-seventh', 'four-eighth', 'four-ninth', 'four-tenth', 'five-third', 'five-quarter', 'five-forth', 'five-fifth', 'five-sixth', 'five-seventh', 'five-eighth', 'five-ninth', 'five-tenth', 'six-third', 'six-quarter', 'six-forth', 'six-fifth', 'six-sixth', 'six-seventh', 'six-eighth', 'six-ninth', 'six-tenth', 'seven-third', 'seven-quarter', 'seven-forth', 'seven-fifth', 'seven-sixth', 'seven-seventh', 'seven-eighth', 'seven-ninth', 'seven-tenth', 'eight-third', 'eight-quarter', 'eight-forth', 'eight-fifth', 'eight-sixth', 'eight-seventh', 'eight-eighth', 'eight-ninth', 'eight-tenth', 'nine-third', 'nine-quarter', 'nine-forth', 'nine-fifth', 'nine-sixth', 'nine-seventh', 'nine-eighth', 'nine-ninth', 'nine-tenth']
    fraction={
        'one-third':1/3,'one-thirds':1/3,'one-quarter':1/4,'one-forth':1/4,'one-fourth':1/4,'one-fourths':1/4,'one-fifth':1/5, 'one-sixth':1/6, 'one-seventh':1/7, 'one-eighth':1/8, 'one-ninth':1/9, 'one-tenth':1/10,'one-fifths':1/5, 'one-sixths':1/6, 'one-sevenths':1/7, 'one-eighths':1/8, 'one-ninths':1/9, 'one-tenths':1/10,\
        'two-third':2/3,'two-thirds':2/3, 'two-quarter':2/4, 'two-forth':2/4,'two-fourth':2/4,'two-fourths':2/4, 'two-fifth':2/5, 'two-sixth':2/6, 'two-seventh':2/7, 'two-eighth':2/8, 'two-ninth':2/9, 'two-tenth':2/10,'two-fifths':2/5, 'two-sixths':2/6, 'two-sevenths':2/7, 'two-eighths':2/8, 'two-ninths':2/9, 'two-tenths':2/10,\
        'three-third':3/3,'three-thirds':3/3, 'three-quarter':3/4, 'three-forth':3/4,'three-fourth':3/4,'three-fourths':3/4, 'three-fifth':3/5, 'three-sixth':3/6, 'three-seventh':3/7, 'three-eighth':3/8, 'three-ninth':3/9, 'three-tenth':3/10,'three-fifths':3/5, 'three-sixths':3/6, 'three-sevenths':3/7, 'three-eighths':3/8, 'three-ninths':3/9, 'three-tenths':3/10,\
        'four-third':4/3,'four-thirds':4/3, 'four-quarter':4/4, 'four-forth':4/4,'four-fourth':4/4,'four-fourths':4/4, 'four-fifth':4/5, 'four-sixth':4/6, 'four-seventh':4/7, 'four-eighth':4/8, 'four-ninth':4/9, 'four-tenth':4/10,'four-fifths':4/5, 'four-sixths':4/6, 'four-sevenths':4/7, 'four-eighths':4/8, 'four-ninths':4/9, 'four-tenths':4/10,\
        'five-third':5/3,'five-thirds':5/3, 'five-quarter':5/4, 'five-forth':5/4,'five-fourth':5/4,'five-fourths':5/4, 'five-fifth':5/5, 'five-sixth':5/6, 'five-seventh':5/7, 'five-eighth':5/8, 'five-ninth':5/9, 'five-tenth':5/10,'five-fifths':5/5, 'five-sixths':5/6, 'five-sevenths':5/7, 'five-eighths':5/8, 'five-ninths':5/9, 'five-tenths':5/10,\
        'six-third':6/3,'six-thirds':6/3, 'six-quarter':6/4, 'six-forth':6/4,'six-fourth':6/4,'six-fourths':6/4, 'six-fifth':6/5, 'six-sixth':6/6, 'six-seventh':6/7, 'six-eighth':6/8, 'six-ninth':6/9, 'six-tenth':6/10,'six-fifths':6/5, 'six-sixths':6/6, 'six-sevenths':6/7, 'six-eighths':6/8, 'six-ninths':6/9, 'six-tenths':6/10,\
        'seven-third':7/3,'seven-thirds':7/3,'seven-quarter':7/4, 'seven-forth':7/4,'seven-fourth':7/4,'seven-fourths':7/4, 'seven-fifth':7/5, 'seven-sixth':7/6, 'seven-seventh':7/7, 'seven-eighth':7/8, 'seven-ninth':7/9, 'seven-tenth':7/10,'seven-fifths':7/5, 'seven-sixths':7/6, 'seven-sevenths':7/7, 'seven-eighths':7/8, 'seven-ninths':7/9, 'seven-tenths':7/10,\
        'eight-third':8/3,'eight-thirds':8/3, 'eight-quarter':8/4, 'eight-forth':8/4,'eight-fourth':8/4,'eight-fourths':8/4, 'eight-fifth':8/5, 'eight-sixth':8/6, 'eight-seventh':8/7, 'eight-eighth':8/8, 'eight-ninth':8/9, 'eight-tenth':8/10,'eight-fifths':8/5, 'eight-sixths':8/6, 'eight-sevenths':8/7, 'eight-eighths':8/8, 'eight-ninths':8/9, 'eight-tenths':8/10,\
        'nine-third':9/3,'nine-thirds':9/3, 'nine-quarter':9/4, 'nine-forth':9/4,'nine-fourth':9/4,'nine-fourths':9/4, 'nine-fifth':9/5, 'nine-sixth':9/6, 'nine-seventh':9/7, 'nine-eighth':9/8, 'nine-ninth':9/9, 'nine-tenth':9/10,'nine-fifths':9/5, 'nine-sixths':9/6, 'nine-sevenths':9/7, 'nine-eighths':9/8, 'nine-ninths':9/9, 'nine-tenths':9/10
    }
    return fraction[number_sentence.lower()]
def english_word_2_num(sentence_list,fraction_acc=None):
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
            elif len(stack)==1 and stack[0].lower() == 'one':
                new_list.append(stack[0])
            elif len(stack)==2  and stack[0].lower() == 'one' and stack[1] == 'point':
                new_list.append(stack[0])
                new_list.append(stack[1])
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
    if fraction_acc!=None:
        num1=['one','two','three','four','five','six','seven','eight','nine']
        num2=['third','thirds','quarter','forth','fourth','fourths','fifth','sixth','seventh','eighth','ninth','tenth','fifths','sixths','sevenths','eighths','ninths','tenths']
        match_word=[]
        for n1 in num1:
            for n2 in num2:
                match_word.append(n1+'-'+n2)
        sentence_list=copy.deepcopy(new_list)
        new_list=[]
        for idx,word in enumerate(sentence_list):
            if word.lower() in match_word :
                number=word_to_num(word)
                number=int(number*10**fraction_acc)/10**fraction_acc
                #number=round(number,fraction_acc)
                new_list.append(str(number))
            else:
                new_list.append(word)
    return new_list


