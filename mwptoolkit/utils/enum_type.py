OPERATORS = ["+", "-", "*", "/", "^"]
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<BRG>", "<OPT>"]
OUTPUT_SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN_IDX = 0


class Operators:
    Single = ["+", "-", "*", "/", "^"]
    Multi = ["+", "-", "*", "/", "^", "=", "<BRG>"]


class FixType:
    Prefix = "prefix"
    Postfix = "postfix"
    Nonfix = None
    MultiWayTree = "multi_way_tree"


class DatasetName:
    math23k = "math23k"
    hmwp = "hmwp"
    mawps = "mawps"
    ape200k = "ape200k"
    alg514 = "alg514"
    draw = "draw"
    SVAMP = "SVAMP"


class DatasetType:
    Train = "train"
    Test = "test"
    Valid = "valid"

class DatasetLanguage:
    en="en"
    zh="zh"

class TaskType:
    MultiEquation = "multi_equation"
    SingleEquation = "single_equation"


class SpecialTokens:
    PAD_TOKEN = "<PAD>" # padding token
    UNK_TOKEN = "<UNK>" # unknown token
    SOS_TOKEN = "<SOS>" # start token
    EOS_TOKEN = "<EOS>" # end token
    NON_TOKEN = "<NON>" # non-terminal token
    BRG_TOKEN = "<BRG>" # equation connecting token
    OPT_TOKEN = "<OPT>" # operator mask token


class MaskSymbol:
    NUM = "NUM"
    alphabet = "alphabet"
    number = "number"


class NumMask:
    NUM = ["NUM"]*100
    alphabet = [
        "NUM_a", "NUM_b", "NUM_c", "NUM_d", "NUM_e", "NUM_f", "NUM_g", "NUM_h", "NUM_i", "NUM_j", "NUM_k", "NUM_l", "NUM_m", "NUM_n", "NUM_o", "NUM_p", "NUM_q", "NUM_r", "NUM_s", "NUM_t", "NUM_u",
        "NUM_v", "NUM_w", "NUM_x", "NUM_y", "NUM_z"
    ]
    number = ["NUM_" + str(i) for i in range(100)]

class SupervisingMode:
    fully_supervised="fully_supervised"
    weakly_supervised="weakly_supervised"
