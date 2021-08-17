<div align='center'><font size='70'>MWPToolkit</font></div>

MWPToolkit is a Python-based framework for math word problem, including popular datasets of math word problem, validated deep learning models and evaluators for different task.

# Environment
python >= 3.6.0

pytorch >= 1.5.0

transformers == 4.3.3

stanza >= 1.2

sympy >= 1.6

ray >= 1.3.0

nltk >= 3.5

gensim >= 3.8.3

word2number >= 1.1

pyltp >= 0.2.1 (optional)

# Supported datasets
<table>
    <tr>
        <td rowspan="4">single-equation dataset</td>
        <td>math23k</td>
    </tr>
    <tr>
        <td>asdiv-a</td>
    </tr>
    <tr>
        <td>mawps-single</td>
    </tr>
    <tr>
        <td>mawps_asdiv-a_svamp</td>
    </tr>
    <tr>
        <td rowspan="4">multiple-equation dataset</td>
        <td>alg514</td>
    </tr>
    <tr>
        <td>draw</td>
    </tr>
    <tr>
        <td>mawps</td>
    </tr>
    <tr>
        <td>hmwp</td>
    </tr>
</table>

# Implemented models
<table>
    <tr>
        <td rowspan="7">Seq2Seq</td>
        <td>DNS</td>
    </tr>
    <tr>
        <td>MathEN</td>
    </tr>
    <tr>
        <td>Saligned</td>
    </tr>
    <tr>
        <td>GroupATT</td>
    </tr>
    <tr>
        <td>EPT</td>
    </tr>
    <tr>
        <td>RNN</td>
    </tr>
    <tr>
        <td>Transformer</td>
    </tr>
    <tr>
        <td rowspan="4">Seq2Tree</td>
        <td>TRNN</td>
    </tr>
    <tr>
        <td>TreeLSTM</td>
    </tr>
    <tr>
        <td>GTS</td>
    </tr>
    <tr>
        <td>SAUSolver</td>
    </tr>
    <tr>
        <td rowspan="3">Graph2Tree</td>
        <td>graph2tree</td>
    </tr>
    <tr>
        <td>MultiE&D</td>
    </tr>
    <tr>
        <td>TSN</td>
    </tr>
    <tr>
        <td rowspan="1">VAE</td>
        <td>RNNVAE</td>
    </tr>
    <tr>
        <td rowspan="3">PreTrain</td>
        <td>BertGen</td>
    </tr>
    <tr>
        <td>RobertaGen</td>
    </tr>
    <tr>
        <td>GPT-2</td>
    </tr>
</table>

# Quick start
## example to run a dataset on a model
```cd MWPToolkit```

```python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation --equation_fix=prefix --k_fold=5 --test_step=5 --gpu_id=0```

## example to run hyper-parameters search

```python run_hyper_search.py --model=Transformer --dataset=math23k --equation_fix=None --task_type=single_equation --k_fold=5 --cpu_per_trial=2 --gpu_per_trial=0.5 --samples=1 --search_file=search_space/Transformer.json --gpu_id=11 ```