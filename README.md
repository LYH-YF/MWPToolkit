<div align='center'><font size='200'>MWPToolkit</font></div>

MWPToolkit is a Python-based framework for math word problem, including popular datasets of math word problem, validated deep learning models and evaluators for different tasks.

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

We hava deployed eight popular datasets in MWP task, these datasets are divided into two categories, single-equation dataset and multiple-equation dataset.

See the table below.

<table align="center">
    <thread align="center">
        <tr>
            <th align="center">task</th>
            <th align="center">dataset</th>
            <th align="center">citation</th>
        </tr>
    </thread>
    <tbody align="center">
        <tr>
            <td rowspan="4" align="center">single-equation dataset</td>
            <td align="center">math23k</td>
            <td align="center"><a href="https://aclanthology.org/D17-1088/">(Wang et al., 2017)</a></td>
        </tr>
        <tr>
            <td align="center">asdiv-a</td>
            <td align="center"><a href="https://aclanthology.org/2020.acl-main.92/">(Miao et al., 2020)</a></td>
        </tr>
        <tr>
            <td align="center">mawps-single</td>
            <td align="center"><a href="https://aclanthology.org/N16-1136/">(Kedziorski et al., 2016)</a></td>
        </tr>
        <tr>
            <td align="center">mawps_asdiv-a_svamp</td>
            <td align="center"><a href="https://arxiv.org/abs/2103.07191">(Patel et al., 2021)</a></td>
        </tr>
        <tr>
            <td rowspan="4" align="center">multiple-equation dataset</td>
            <td align="center">alg514</td>
            <td align="center"><a href="https://aclanthology.org/P14-1026/">(Kushman et al., 2014)</a></td>
        </tr>
        <tr>
            <td align="center">draw</td>
            <td align="center"><a href="https://arxiv.org/abs/1609.07197">(Upadhyay et al., 2017)</a></td>
        </tr>
        <tr>
            <td align="center">mawps</td>
            <td align="center"><a href="https://aclanthology.org/N16-1136/">(Kedziorski et al., 2016)</a></td>
        </tr>
        <tr>
            <td align="center">hmwp</td>
            <td align="center"><a href="https://arxiv.org/abs/2010.06823">(Qin et al., 2020)</a></td>
        </tr>
    </tbody>
</table>
Other popular datasets like ape200k<a href="https://arxiv.org/abs/2009.11506">(Zhao et al., 2020)</a>, dolphin1878<a href="https://aclanthology.org/D15-1135/">(Shi et al., 2015)</a> and dolphin18k<a href="https://aclanthology.org/P16-1084/">(Huang et al., 2016)</a> we will finish deployment soon. 

# Models
We have implemented 18 deep learning models and we are updating some other models. 

See two tables below.

# Implemented models
<table align="center">
    <thread>
        <tr>
            <th align="center">type</th>
            <th align="center">model</th>
            <th align="center">citation</th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td rowspan="7">Seq2Seq</td>
            <td>DNS</td>
            <td align="center"><a href="https://aclanthology.org/D17-1088/">(Wang et al., 2017)</a></td>
        </tr>
        <tr>
            <td>MathEN</td>
            <td align="center"><a href="https://aclanthology.org/D18-1132/">(Wang et al., 2018)</a></td>
        </tr>
        <tr>
            <td>Saligned</td>
            <td align="center"><a href="https://aclanthology.org/N19-1272/">(Chiang et al., 2019)</a></td>
        </tr>
        <tr>
            <td>GroupATT</td>
            <td align="center"><a href="https://aclanthology.org/P19-1619/">(Li et al., 2019)</a></td>
        </tr>
        <tr>
            <td>EPT</td>
            <td align="center"><a href="https://aclanthology.org/2020.emnlp-main.308/">(Kim et al., 2020)</a></td>
        </tr>
        <tr>
            <td>RNN</td>
            <td align="center"><a href="https://arxiv.org/abs/1409.3215">(Sutskever et al., 2014)</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1706.03762">(Vaswani et al., 2017)</a></td>
        </tr>
        <tr>
            <td rowspan="4">Seq2Tree</td>
            <td>TRNN</td>
            <td align="center"><a href="https://ojs.aaai.org//index.php/AAAI/article/view/4697">(Wang et al., 2019)</a></td>
        </tr>
        <tr>
            <td>TreeLSTM</td>
            <td align="center"><a href="https://aclanthology.org/D19-1241/">(Liu et al., 201*)</a></td>
        </tr>
        <tr>
            <td>GTS</td>
            <td align="center"><a href="https://www.ijcai.org/proceedings/2019/736">(Xie et al., 2019)</a></td>
        </tr>
        <tr>
            <td>SAUSolver</td>
            <td align="center"><a href="https://arxiv.org/abs/2010.06823">(Qin et al., 2020)</a></td>
        </tr>
        <tr>
            <td rowspan="3">Graph2Tree</td>
            <td>graph2tree</td>
            <td align="center"><a href="https://aclanthology.org/2020.acl-main.362/">(Zhang et al., 2020)</a></td>
        </tr>
        <tr>
            <td>MultiE&D</td>
            <td align="center"><a href="https://aclanthology.org/2020.coling-main.262/">(Shen et al., 2020)</a></td>
        </tr>
        <tr>
            <td>TSN</td>
            <td align="center"><a href="https://www.ijcai.org/proceedings/2020/555">(Zhang et al., 2020)</a></td>
        </tr>
        <tr>
            <td rowspan="1">VAE</td>
            <td>RNNVAE</td>
            <td align="center"><a href="https://arxiv.org/abs/1605.07869">(Zhang et al., 2016)</a></td>
        </tr>
        <tr>
            <td rowspan="3">PreTrain</td>
            <td>BertGen</td>
            <td align="center"><a href="https://arxiv.org/abs/1810.04805">(Devlin et al., 2018)</a></td>
        </tr>
        <tr>
            <td>RobertaGen</td>
            <td align="center"><a href="https://arxiv.org/abs/1907.11692">(Liu et al., 2019)</a></td>
        </tr>
        <tr>
            <td>GPT-2</td>
            <td align="center"><a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">(Radford et al., 2019)</a></td>
        </tr>
    </tbody>
</table>

## Updating models
<table align="center">
    <thread>
        <tr>
            <th align="center">model</th>
            <th align="center">citation</th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td>KA-S2T</td>
            <td align="center"><a href="https://aclanthology.org/2020.emnlp-main.579/">(Wu et al., 2020)</a></td>
        </tr>
        <tr>
            <td>HMS</td>
            <td align="center"><a href="https://ojs.aaai.org/index.php/AAAI/article/view/16547">(Lin et al., 2021)</a></td>
        </tr>
        <tr>
            <td>NUM-S2T</td>
            <td align="center"><a href="https://aclanthology.org/2021.acl-long.455/">(Wu et al., 2021)</a></td>
        </tr>
    </tbody>
</table>


# Quick start
```cd MWPToolkit```
## 1.Example to run a dataset on a model

```python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation --equation_fix=prefix --k_fold=5 --test_step=5 --gpu_id=0```

### Parameters
 ```model``` and ```dataset``` are the model and dataset you  specify to run,

```task_type```, the value of it should be in [single_equation | multi_equation]. Usually, it's up to dataset.

```equation_fix```,the value of it should be in [infix | postfix | prefix]. it decides the form of output of the model. Some models requires specific form of the equation, so set this parameter to avoid bad performance because of the incorrect output form.

```k_fold```, it decides whether to run k-fold cross validation on the dataset, if you don't set this parameter, and then it will run train-valid-test split procedure. 

```test_step```, you can set a integer to decide how many epochs to test the model's performance.

```gpu_id```, if you want to train model with GPU, you can set this parameter to specify GPU. Note that we haven't tested the framework with multipul GPUs yet.

## 2.Run a new dataset
Our supported datasets are all saved under the folder ```'dataset'```.
If you want to run your own dataset, you can follow the steps below.

<strong>First</strong>:

Your dataset folder (same as the dataset name) should include three json files: 

```
dataset_name
    |----trainset.json
    |----validset.json
    |----testset.json
```

Move your dataset folder to ```'dataset'``` of our framework, then the file structure will be like:

```
dataset
    |----dataset_name
            |----trainset.json
            |----validset.json
            |----testset.json
```

<strong>Second</strong>:

Set your dataset configuration, our supported dataset configuration files are saved at ```'mwptoolkit/properties/dataset/'```. You can create a json file under the folder. The road path will be like:

```mwptoolkit/properties/dataset/dataset_name.json```

<strong>Finaly</strong>:

Run:

```python run_mwptoolkit.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --gpu_id=0```

### Parameters
If you don't move your dataset folder and dataset configuration file to specific folder,these parameters can be set directly.

```dataset_path```, the default value is like ```dataset/dataset_name```, you can set your own dataset road path ```--dataset_path=[your_dataset]``` in cmd line.

```dataset_config_path```, the default value is like ```mwptoolkit/properties/dataset/dataset_name.json```, you can set your own dataset configuration file ```--dataset_config_path=[your_dataset_configuration]``` in cmd line.

## example to run hyper-parameters search

We implemented hyper-parameter search in our framework based ```ray.tune```.

You can run cmd line template below:

```python run_hyper_search.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --cpu_per_trial=2 --gpu_per_trial=0.5 --samples=1 --search_file=search_file.json --gpu_id=0```

### Parameters

```cpu_per_trial```, CPU resources to allocate per trial.

```gpu_per_trial```, GPU resources to allocate per trial.

```samples```, times to sample from the search space.

```search_file```, a json file including search parameter name and space.

```search_parameter```, if you don't write the search file, you can set this parameter to specify search space, e.g. ```--search_parameter=hidden_size=[256,512] --search_parameter=embedding_size=[64,128,256] --search_parameter=learning_rate='(1e-4, 1e-2)' ```