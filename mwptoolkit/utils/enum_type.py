# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:15:02
# @File: enum_type.py


from itertools import groupby
from math import log10
from sympy import Eq
import re

OPERATORS = ["+", "-", "*", "/", "^"]
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<BRG>", "<OPT>"]
OUTPUT_SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN_IDX = 0


class Operators:
    """operators in equation.
    """
    Single = ["+", "-", "*", "/", "^"]
    Multi = ["+", "-", "*", "/", "^", "=", "<BRG>"]


class FixType:
    """equation fix type
    """
    Prefix = "prefix"
    Postfix = "postfix"
    Infix = "infix"
    Nonfix = None
    MultiWayTree = "multi_way_tree"


class DatasetName:
    """dataset name
    """
    math23k = "math23k"
    hmwp = "hmwp"
    mawps = "mawps"
    ape200k = "ape200k"
    alg514 = "alg514"
    draw = "draw"
    SVAMP = "SVAMP"
    asdiv_a = "asdiv-a"
    mawps_single = "mawps-single"
    mawps_asdiv_a_svamp = "mawps_asdiv-a_svamp"


class DatasetType:
    """dataset type
    """
    Train = "train"
    Test = "test"
    Valid = "valid"

class DatasetLanguage:
    """dataset language
    """
    en="en"
    zh="zh"

class TaskType:
    """task type
    """
    MultiEquation = "multi_equation"
    SingleEquation = "single_equation"


class SpecialTokens:
    """special tokens
    """
    PAD_TOKEN = "<PAD>" # padding token
    UNK_TOKEN = "<UNK>" # unknown token
    SOS_TOKEN = "<SOS>" # start token
    EOS_TOKEN = "<EOS>" # end token
    NON_TOKEN = "<NON>" # non-terminal token
    BRG_TOKEN = "<BRG>" # equation connecting token
    OPT_TOKEN = "<OPT>" # operator mask token


class MaskSymbol:
    """number mask type
    """
    NUM = "NUM"
    alphabet = "alphabet"
    number = "number"


class NumMask:
    """number mask symbol list
    """
    NUM = ["NUM"]*100
    alphabet = [
        "NUM_a", "NUM_b", "NUM_c", "NUM_d", "NUM_e", "NUM_f", "NUM_g", "NUM_h", "NUM_i", "NUM_j", "NUM_k", "NUM_l", "NUM_m", "NUM_n", "NUM_o", "NUM_p", "NUM_q", "NUM_r", "NUM_s", "NUM_t", "NUM_u",
        "NUM_v", "NUM_w", "NUM_x", "NUM_y", "NUM_z"
    ]
    number = ["NUM_" + str(i) for i in range(100)]

class SupervisingMode:
    """supervising mode"""
    fully_supervised="fully_supervised"
    weakly_supervised=["fix", "mafix", "reinforce", "mapo"]

class EPT:
    FRACTIONAL_PATTERN = re.compile('(\\d+/\\d+)')
    # Pattern of numbers e.g. 2,930.34
    NUMBER_PATTERN = re.compile('([+\\-]?(\\d{1,3}(,\\d{3})+|\\d+)(\\.\\d+)?)')
    # Pattern of number and fraction numbers
    NUMBER_AND_FRACTION_PATTERN = re.compile('(%s|%s)' % (FRACTIONAL_PATTERN.pattern, NUMBER_PATTERN.pattern))
    # Pattern of numbers that following zeros under the decimal point. e.g., 0_250000000
    FOLLOWING_ZERO_PATTERN = re.compile('(\\d+|\\d+_[0-9]*[1-9])_?(0+|0{4}\\d+)$')
    MULTIPLES = ['once', 'twice', 'thrice', 'double', 'triple', 'quadruple', 'dozen', 'half', 'quarter',
                'doubled', 'tripled', 'quadrupled', 'halved', 'quartered']

    # Suffix of plural forms
    PLURAL_FORMS = [('ies', 'y'), ('ves', 'f'), ('s', '')]
    NUMBER_READINGS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000,

    'once': 1, 'twice': 2, 'thrice': 3, 'double': 2, 'triple': 3, 'quadruple': 4,
    'doubled': 2, 'tripled': 3, 'quadrupled': 4,

    'third': 3, 'forth': 4, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13,
    'fourteenth': 14, 'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19,
    'twentieth': 20, 'thirtieth': 30, 'fortieth': 40, 'fiftieth': 50, 'sixtieth': 60,
    'seventieth': 70, 'eightieth': 80, 'ninetieth': 90,
    'hundredth': 100, 'thousandth': 1000, 'millionth': 1000000,
    'billionth': 1000000000,

    'dozen': 12, 'half': 0.5, 'quarter': 0.25,
    'halved': 0.5, 'quartered': 0.25,
    }

    OPERATOR_PRECEDENCE = {
    '^': 4,
    '*': 3,
    '/': 3,
    '+': 2,
    '-': 2,
    '=': 1
    }
    PAD_ID = -1

    # Key indices for preprocessing and field input
    PREP_KEY_EQN = 0
    PREP_KEY_ANS = 1
    PREP_KEY_MEM = 2

    # Token for text field
    NUM_TOKEN = '[N]'
    SPIECE_UNDERLINE = '▁'

    # String key names for inputs
    IN_TXT = 'text'
    IN_TPAD = 'text_pad'
    IN_TNUM = 'text_num'
    IN_TNPAD = 'text_numpad'
    IN_EQN = 'equation'

    # Dictionary of operators
    OPERATORS = {
        '+': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] + x[1])},
        '-': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] - x[1])},
        '*': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] * x[1])},
        '/': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] / x[1])},
        '^': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] ** x[1])},
        '=': {'arity': 2, 'commutable': True, 'top_level': True,
              'convert': (lambda *x: Eq(x[0], x[1], evaluate=False))}
    }

    # Arity and top-level classes
    TOP_LEVEL_CLASSES = ['Eq']
    ARITY_MAP = {key: [item[-1] for item in lst]
                 for key, lst in
                 groupby(sorted([((op['arity'], op['top_level']), key) for key, op in OPERATORS.items()],
                                key=lambda t: t[0]), key=lambda t: t[0])}

    # Infinity values
    NEG_INF = float('-inf')
    POS_INF = float('inf')

    # FOR EXPRESSION INPUT
    # Token for operator field
    FUN_NEW_EQN = '__NEW_EQN'
    FUN_END_EQN = '__DONE'
    FUN_NEW_VAR = '__NEW_VAR'
    FUN_TOKENS = [FUN_NEW_EQN, FUN_END_EQN, FUN_NEW_VAR]
    FUN_NEW_EQN_ID = FUN_TOKENS.index(FUN_NEW_EQN)
    FUN_END_EQN_ID = FUN_TOKENS.index(FUN_END_EQN)
    FUN_NEW_VAR_ID = FUN_TOKENS.index(FUN_NEW_VAR)

    FUN_TOKENS_WITH_EQ = FUN_TOKENS + ['=']
    FUN_EQ_SGN_ID = FUN_TOKENS_WITH_EQ.index('=')

    # Token for operand field
    ARG_CON = 'CONST:'
    ARG_NUM = 'NUMBER:'
    ARG_MEM = 'MEMORY:'
    ARG_TOKENS = [ARG_CON, ARG_NUM, ARG_MEM]
    ARG_CON_ID = ARG_TOKENS.index(ARG_CON)
    ARG_NUM_ID = ARG_TOKENS.index(ARG_NUM)
    ARG_MEM_ID = ARG_TOKENS.index(ARG_MEM)
    ARG_UNK = 'UNK'
    ARG_UNK_ID = 0

    # Maximum capacity of variable, numbers and expression memories
    VAR_MAX = 2
    NUM_MAX = 32
    MEM_MAX = 32

    # FOR OP INPUT
    SEQ_NEW_EQN = FUN_NEW_EQN
    SEQ_END_EQN = FUN_END_EQN
    SEQ_UNK_TOK = ARG_UNK
    SEQ_TOKENS = [SEQ_NEW_EQN, SEQ_END_EQN, SEQ_UNK_TOK, '=']
    SEQ_PTR_NUM = '__NUM'
    SEQ_PTR_VAR = '__VAR'
    SEQ_PTR_TOKENS = SEQ_TOKENS + [SEQ_PTR_NUM, SEQ_PTR_VAR]
    SEQ_NEW_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_NEW_EQN)
    SEQ_END_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_END_EQN)
    SEQ_UNK_TOK_ID = SEQ_PTR_TOKENS.index(SEQ_UNK_TOK)
    SEQ_EQ_SGN_ID = SEQ_PTR_TOKENS.index('=')
    SEQ_PTR_NUM_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_NUM)
    SEQ_PTR_VAR_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_VAR)
    SEQ_GEN_NUM_ID = SEQ_PTR_NUM_ID
    SEQ_GEN_VAR_ID = SEQ_GEN_NUM_ID + NUM_MAX

    # Format of variable/number/expression tokens
    FORMAT_VAR = 'X_%%0%dd' % (int(log10(VAR_MAX)) + 1)
    FORMAT_NUM = 'N_%%0%dd' % (int(log10(NUM_MAX)) + 1)
    FORMAT_MEM = 'M_%%0%dd' % (int(log10(MEM_MAX)) + 1)
    VAR_PREFIX = 'X_'
    NUM_PREFIX = 'N_'
    CON_PREFIX = 'C_'
    MEM_PREFIX = 'M_'

    # Key for field names
    FIELD_OP_GEN = 'op_gen'
    FIELD_EXPR_GEN = 'expr_gen'
    FIELD_EXPR_PTR = 'expr_ptr'

    # Model names
    MODEL_VANILLA_TRANS = 'vanilla'  # Vanilla Op Transformer
    MODEL_EXPR_TRANS = 'expr'  # Vanilla Transformer + Expression (Expression Transformer)
    MODEL_EXPR_PTR_TRANS = 'ept'

    