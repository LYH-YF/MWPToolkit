def joint_number_(text_list): #match longer fraction such as ( 1 / 1000000 )
    #text_list='( | ( 5 / 7 ) - ( 1 / 14 ) + ( 5 / 6 ) ) / ( 5 / 42 ) ．'.split(" ")
    #text_list="计 ( 123 (1/10000) ) 算 ( 1 / 2 ) + ( 1 / 6 ) + ( 1 / 12 ) + ( 1 / 20 ) + … + ( 1 / 380 ) = 多 少 ．".split()
    text_list="苹 果 树 比 梨 树 少 ( 3 / 8 ) ， 梨 树 比 苹 果 树 多 ( ( ( ) ) / ( ( ) ) )".split(" ")
    new_list=[]
    i=0
    while i < len(text_list):
        if text_list[i] == '(':
            try:
                j=text_list[i:].index(')')
                if i+1==i+j:
                    j=None
                if "(" in text_list[i+1:i+j+1]:
                    j=None
            except:
                j=None
            if j:
                stack=[]
                flag=True
                idx=0
                for temp_idx,word in enumerate(text_list[i:i+j+1]):
                    if word in ["(",")","/"] or word.isdigit():
                        stack.append(word)
                        idx=temp_idx
                    else:
                        flag=False
                        break
                if "/" not in stack:
                    flag=False
                if flag:
                    number=''.join(stack)
                    new_list.append(number)
                else:
                    for word in stack:
                        new_list.append(word)
                i+=idx+1
            else:
                new_list.append(text_list[i])
                i+=1
        else:
            new_list.append(text_list[i])
            i+=1
    return new_list
#joint_number_([])

