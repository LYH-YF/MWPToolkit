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
{
    "original": {
        "sQuestion": "mary has 7 more than twice as many quarters as dimes . Her total is $ 10.15 , how many of each coin did she have ?", 
        "lSolutions": [21.0, 49.0], 
        "Template": ["m - a * n = b", "c * n + d * m = e"], 
        "lEquations": ["7+2*x=y", "0.25*x+0.1*y=10.15"], 
        "iIndex": 659592, 
        "Alignment": [
            {"coeff": "a", "SentenceId": 0, "Value": 2.0, "TokenId": 5}, 
            {"coeff": "b", "SentenceId": 0, "Value": 7.0, "TokenId": 2}, 
            {"coeff": "c", "SentenceId": 0, "Value": 0.25, "TokenId": 8}, 
            {"coeff": "d", "SentenceId": 0, "Value": 0.10000000149011612, "TokenId": 10}, 
            {"coeff": "e", "SentenceId": 1, "Value": 10.15, "TokenId": 4}], 
            "Equiv": []}, 
        "text": "mary has 7 more than twice as many quarters as dimes . Her total is $ 10.15 , how many of each coin did she have ?", 
        "numbers": [
            {"token": [2], "value": "7", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, 
            {"token": [5], "value": "2", "is_text": true, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": true, "is_combined_multiple": false}, 
            {"token": [8], "value": "0.25000000000000", "is_text": true, "is_integer": false, "is_ordinal": false, "is_fraction": false, "is_single_multiple": true, "is_combined_multiple": false}, 
            {"token": [16], "value": "10.15000000000000", "is_text": false, "is_integer": false, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}], 
        "answer": [[21.0, 49.0]], 
        "expr": [[0, "X_0 N_1 X_1 * - N_0 ="], [0, "N_2 X_1 * C_0_1 X_0 * + N_3 ="], [1, "X_0 X_1"]], 
        "id": "draw_train-659592", 
        "set": "draw_train"}
