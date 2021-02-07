# MWPToolkit
A toolkit for math word problem.

## quick start
example:
    run "python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation"

## experiment
|model     |value acc |equ acc   |
|----------|----------|----------|
|GTS       |0.73+     |-         |
|RNN       |0.674     |0.578     |
|Transformer|        -|         -|
|RNNVAE    |0.713     |0.607     |
