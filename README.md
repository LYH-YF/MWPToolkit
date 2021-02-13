# MWPToolkit
A toolkit for math word problem.

## quick start
example:
    run "python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation"

## experiment
|model               |value acc |equ acc   |
|--------------------|----------|----------|
|GTS                 |0.73+     |-         |
|RNN                 |0.654     |0.562     |
|RNN(prefix)         |0.674     |0.578     |
|RNN(postfix)        |0.66      |0.568     |
|Transformer         |         -|         -|
|RNNVAE(prefix)      |0.713     |0.607     |
