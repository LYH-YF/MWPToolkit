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
    return new_list


