import re

from mwptoolkit.utils.utils import str2float
from mwptoolkit.utils.enum_type import EPT
from mwptoolkit.utils.preprocess_tool.sentence_operator import find_ept_numbers_in_text
from mwptoolkit.utils.preprocess_tool.number_operator import constant_number
from mwptoolkit.utils.preprocess_tool.equation_operator import orig_infix_to_postfix,infix_to_postfix


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


def preprocess_ept_dataset_(train_datas, valid_datas, test_datas, dataset_name):
    train_datas = ept_preprocess(train_datas, dataset_name)
    valid_datas = ept_preprocess(valid_datas, dataset_name)
    test_datas = ept_preprocess(test_datas, dataset_name)
    return train_datas, valid_datas, test_datas

def ept_preprocess(datas, dataset_name):
    datas_list = []
    
    
    for idx, data in enumerate(datas):
        if dataset_name == "mawps":
            
            answer_list = [(x,) for x in data['aux']['lSolutions']]
            masked_text = re.sub('\\s+', ' ', data['aux']['mask_text']).strip().split(' ')
            temp_tokens = data['aux']['num_list']

            regenerated_text = []
            for token in masked_text:
                if token.startswith('temp_'):
                    regenerated_text.append(str(temp_tokens[int(token[5:])]))
                else:
                    regenerated_text.append(token)

            problem = ' '.join(regenerated_text)
        elif dataset_name == "SVAMP":
            data["original_text"] = data["ques source 1"].strip()
            data["ans"] = [str2float(data["Answer"])]
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()
        elif dataset_name == "asdiv-a":
            data["original_text"] = data["ques source 1"].strip()
            if 'r' in data["ans"]:
                data["ans"] = data["ans"][:2]
            data["ans"] = [str2float(data["ans"])]
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()
        elif dataset_name == "mawps_asdiv-a_svamp":
            data["original_text"] = data["ques source 1"].strip()
            data['ans']=[data['ans']]
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()
        elif dataset_name == 'math23k':
            data["original_text"] = data["ques source 1"].strip()
            data["ans"] = [str2float(data["ans"])]
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()
            #if '^' in data['infix equation']:
            #    continue
        elif dataset_name == 'hmwp':
            data['original_text'] = data['ques source 1']
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()
        elif dataset_name == 'alg514' or dataset_name == 'draw':
            answer_list = [tuple(x for x in data['ans'])]
            problem = data["original_text"].strip()

        text, numbers = find_ept_numbers_in_text(problem)
        data['ept'] = {}
        data['ept']['text'] = text

        data['ept']['numbers'] = numbers
        
        data['ept']['answer'] = answer_list
        prefix_formula = refine_formula_as_prefix(data, numbers, dataset_name)
        data['ept']['expr'] = prefix_formula
        
        datas_list.append(data)
    return datas_list

def refine_formula_as_prefix(item, numbers, dataset_name):
    if dataset_name in ['SVAMP','asdiv-a','math23k','mawps_asdiv-a_svamp']:
        formula = item['infix equation']
        
        formula = ["x", "="]+formula
    else:
        formula = item['infix equation']
    
    if dataset_name in ["alg514", 'draw']:
        formula = [re.sub('([-+*/=])', ' \\1 ', eqn.lower().replace('-1', '1NEG')).replace('1NEG', '-1')
                   for eqn in item["aux"]['Template']]  # Shorthand for linear formula
        tokens = re.split('\\s+', item['aux']['sQuestion'].strip())
        number_by_tokenid = {j: i for i, x in enumerate(numbers) for j in x['token']}

        # Build map between (sentence, token in sentence) --> number token index
        number_token_sentence = {}
        sent_id = 0
        sent_token_id = 0
        for tokid, token in enumerate(tokens):
            if token in '.!?':  # End of sentence
                sent_id += 1
                sent_token_id = 0
                continue

            if tokid in number_by_tokenid:
                number_token_sentence[(sent_id, sent_token_id)] = number_by_tokenid[tokid]

            sent_token_id += 1

        # [1] Build mapping between coefficients in the template and var names (N_0, T_0, ...)
        mappings = {}
        for align in item["aux"]['Alignment']:
            var = align['coeff']
            val = align['Value']
            sent_id = align['SentenceId']
            token_id = align['TokenId']

            if (sent_id, token_id) not in number_token_sentence:
                # If this is not in numbers recognized by our system, regard it as a constant.
                positive, const_code = constant_number(val)
                mappings[var] = [const_code]
                if not positive:
                    mappings[var].append('-')

                continue

            number_id = number_token_sentence[(sent_id, token_id)]
            number_info = numbers[number_id]

            expression = ['N_%s' % number_id]
            expr_value = eval(number_info['value'])

            offset = 1
            while abs(val - expr_value) > 1E-10 and (sent_id, token_id + offset) in number_token_sentence:
                next_number_id = number_token_sentence[(sent_id, token_id + offset)]
                next_info = numbers[next_number_id]
                next_value = eval(next_info['value'])
                next_token = 'N_%s' % next_number_id

                if next_value >= 100:
                    # Multiplicative case: e.g. '[Num] million'
                    expr_value *= next_value
                    # As a postfix expression
                    expression.append(next_token)
                    expression.append('*')
                else:
                    # Additive case: e.g. '[NUM] hundred thirty-two'
                    expr_value += next_value
                    expression.append(next_token)
                    expression.append('+')

                offset += 1

            # Final check.
            # assert abs(val - expr_value) < 1E-5, "%s vs %s: \n%s\n%s" % (align, expr_value, numbers, item)
            mappings[var] = expression

        # [2] Parse template and convert coefficients into our variable names.
        # Free symbols in the template denotes variables representing the answer.
        new_formula = []
        free_symbols = []

        for eqn in formula:
            output_tokens = orig_infix_to_postfix(eqn, mappings, free_symbols)

            if output_tokens:
                new_formula.append((EPT.PREP_KEY_EQN, output_tokens))

        if free_symbols:
            new_formula.append((EPT.PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))
    elif dataset_name in ['mawps']:
        template_to_number = {}
        template_to_value = {}
        
        number_by_tokenid = {j: i for i, x in enumerate(numbers) for j in x['token']}

        for tokid, token in enumerate(re.sub('\\s+', ' ', item['aux']['mask_text']).strip().split(' ')):
            if token.startswith('temp_'):
                assert tokid in number_by_tokenid, (tokid, number_by_tokenid, item['aux'])

                num_id = number_by_tokenid[tokid]
                template_to_number[token] = ['N_%s' % num_id]
                template_to_value[token] = numbers[num_id]['value']

        # We should read both template_equ and new_equation because of NONE in norm_post_equ.
        formula = item['aux']['template_equ'].split(' ')
        original = item['aux']['new_equation'].split(' ')
        assert len(formula) == len(original)

        # Recover 'NONE' constant in the template_equ.
        for i in range(len(formula)):
            f_i = formula[i]
            o_i = original[i]

            if f_i == 'NONE':
                formula[i] = original[i]
            elif f_i.startswith('temp_'):
                assert abs(float(template_to_value[f_i]) - float(o_i)) < 1E-4,\
                    "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)
            else:
                # Check whether two things are the same.
                assert f_i == o_i, "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)

        free_symbols = []
        new_formula = [(EPT.PREP_KEY_EQN, orig_infix_to_postfix(formula, template_to_number, free_symbols))]

        if free_symbols:
            new_formula.append((EPT.PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))
    else:
        for wordid, word in enumerate(formula):
            if word == '[' or word == '{':
                formula[wordid] = '('
            elif word == ']' or word == '}':
                formula[wordid] = ')'
        formula.append("<BRG>")
        formula_list = []
        formula_string = ''
        for word in formula:
            if word == '<BRG>':
                formula_list.append(formula_string.strip())
                formula_string = ''
            else:
                formula_string += word
                formula_string += ' '
        formula = formula_list
        new_formula = []
        free_symbols = []

        for eqn in formula:
            output_tokens = infix_to_postfix(eqn, free_symbols)

            if output_tokens:
                new_formula.append((EPT.PREP_KEY_EQN, output_tokens))

        if free_symbols:
            new_formula.append((EPT.PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))

    return new_formula

