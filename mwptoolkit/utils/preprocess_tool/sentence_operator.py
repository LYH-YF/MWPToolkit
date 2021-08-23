import re

import nltk
import stanza

from mwptoolkit.utils.data_structure import DependencyTree
from mwptoolkit.utils.utils import write_json_data,read_json_data
from mwptoolkit.utils.enum_type import EPT

def deprel_tree_to_file(train_datas, valid_datas, test_datas, path, language, use_gpu):
    """save deprel tree infomation to file
    """
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
    """get group nums infomation from file.
    """
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
                        if n_pos + 1 < sent_len:
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
    """get deprel tree infomation from file
    """
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
    """split sentence by punctuations.
    """
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

def find_ept_numbers_in_text(text: str, append_number_token: bool = False):
    
    numbers = []
    new_text = []

    # Replace "[NON-DIGIT][SPACEs].[DIGIT]" with "[NON-DIGIT][SPACEs]0.[DIGIT]"
    text = re.sub("([^\\d.,]+\\s*)(\\.\\d+)", "\\g<1>0\\g<2>", text)
    # Replace space between digits or digit and special characters like ',.' with "⌒" (to preserve original token id)
    text = re.sub("(\\d+)\\s+(\\.\\d+|,\\d{3}|\\d{3})", "\\1⌒\\2", text)

    # Original token index
    i = 0
    prev_token = None
    for token in text.split(' '):
        # Increase token id and record original token indices
        token_index = [i + j for j in range(token.count('⌒') + 1)]
        i = max(token_index) + 1

        # First, find the number patterns in the token
        token = token.replace('⌒', '')
        number_patterns = EPT.NUMBER_AND_FRACTION_PATTERN.findall(token)
        if number_patterns:
            for pattern in number_patterns:
                # Matched patterns, listed by order of occurrence.
                surface_form = pattern[0]
                surface_form = surface_form.replace(',', '')

                # Normalize the form: use the decimal point representation with 15-th position under the decimal point.
                is_fraction = '/' in surface_form
                value = eval(surface_form)
                if type(value) is float:
                    surface_form = EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=False, is_integer='.' not in surface_form,
                                    is_ordinal=False, is_fraction=is_fraction,
                                    is_single_multiple=False, is_combined_multiple=False))

            new_text.append(EPT.NUMBER_AND_FRACTION_PATTERN.sub(' \\1 %s ' % EPT.NUM_TOKEN, token))
        else:
            # If there is no numbers in the text, then find the textual numbers.
            # Append the token first.
            new_text.append(token)

            # Type indicator
            is_ordinal = False
            is_fraction = False
            is_single_multiple = False
            is_combined_multiple = False

            subtokens = re.split('[^a-zA-Z0-9]+', token.lower())  # Split hypen-concatnated tokens like twenty-three
            token_value = None
            for subtoken in subtokens:
                if not subtoken:
                    continue

                # convert to singular nouns
                for plural, singluar in EPT.PLURAL_FORMS:
                    if subtoken.endswith(plural):
                        subtoken = subtoken[:-len(plural)] + singluar
                        break

                if subtoken in EPT.NUMBER_READINGS:
                    if not token_value:
                        # If this is the first value in the token, then set it as it is.
                        token_value = EPT.NUMBER_READINGS[subtoken]

                        is_ordinal = subtoken[-2:] in ['rd', 'th']
                        is_single_multiple = subtoken in EPT.MULTIPLES

                        if is_ordinal and prev_token == 'a':
                            # Case like 'A third'
                            token_value = 1 / token_value
                    else:
                        # If a value was set before reading this subtoken,
                        # then treat it as multiples. (e.g. one-million, three-fifths, etc.)
                        followed_value = EPT.NUMBER_READINGS[subtoken]
                        is_single_multiple = False
                        is_ordinal = False

                        if followed_value >= 100 or subtoken == 'half':  # case of unit
                            token_value *= followed_value
                            is_combined_multiple = True
                        elif subtoken[-2:] in ['rd', 'th']:  # case of fractions
                            token_value /= followed_value
                            is_fraction = True
                        else:
                            token_value += followed_value

            # If a number is found.
            if token_value is not None:
                if type(token_value) is float:
                    surface_form = EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % token_value)
                else:
                    surface_form = str(token_value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=True, is_integer='.' not in surface_form,
                                    is_ordinal=is_ordinal, is_fraction=is_fraction,
                                    is_single_multiple=is_single_multiple,
                                    is_combined_multiple=is_combined_multiple))
                new_text.append(EPT.NUM_TOKEN)

        prev_token = token

    if append_number_token:
        text = ' '.join(new_text)

    return text, numbers

