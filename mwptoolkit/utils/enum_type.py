OPERATORS = ["+", "-", "*", "/", "^"]
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
OUTPUT_SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN_IDX = 0


class FixType:
    Prefix = "prefix"
    Postfix = "postfix"
    Nonfix = None


class DatasetType:
    Train = "train"
    Test = "test"
    Valid = "valid"


class TaskType:
    MultiEquation = "multi_equation"
    SingleEquation = "single_equation"


class SpecialTokens:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"


class MaskSymbol:
    NUM = "NUM"
    alphabet = "alphabet"
    number = "number"


class NumMask:
    NUM = ["NUM"]
    alphabet = [
        "NUM_a", "NUM_b", "NUM_c", "NUM_d", "NUM_e", "NUM_f", "NUM_g", "NUM_h",
        "NUM_i", "NUM_j", "NUM_k", "NUM_l", "NUM_m", "NUM_n", "NUM_o", "NUM_p",
        "NUM_q", "NUM_r", "NUM_s", "NUM_t", "NUM_u", "NUM_v", "NUM_w", "NUM_x",
        "NUM_y", "NUM_z"
    ]
    number = ["NUM_" + str(i) for i in range(100)]
