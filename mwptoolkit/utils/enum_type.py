OPERATORS=["+","-","*","/","^"]
INPUT_SPECIAL_TOKENS=["<PAD>","<UNK>","<SOS>","<EOS>"]
OUTPUT_SPECIAL_TOKENS=["<PAD>","<UNK>"]
PAD_TOKEN="<PAD>"
UNK_TOKEN="<UNK>"
SOS_TOKEN="<SOS>"
EOS_TOKEN="<EOS>"
PAD_TOKEN_IDX=0

class TaskType:
    MultiEquation="multi_equation"
    SingleEquation="single_equation"