# <div align="center"> MWPToolkit </div>

[Doc]()|[Model]()|[Dataset]()|[Paper]()

MWPToolkit is a PyTorch-based toolkit for Math Word Problem(MWP) solving task. It is a comprehensive and efficient framework which integrates **xx** popular MWP benchmark datasets and **xx** deep learning-based MWP algorithms and evaluators for different measurement metrics. 

Our framework has the following architecture. You could ulitize our toolkit to evaluate the build-in datasets, apply it to process your raw data copies or develop your own models. **(YS: draw a figure about the framework architecture)**

![](https://octodex.github.com/images/yaktocat.png)
<div align="center"> Figure: Architecture of MWP Toolkit </div>

# News

# Feature

# Installation
Development environment **(YS: move other libraries expect python, pytorch, transformers to requirements.txt file)**:
```
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
```

**Method 1: Install from pip**

**(YS: figure out how to make pip install of this libraries)**
```
pip install 
```
**Method 2: Install from source**

**(YS: check if this is correct)**
```
# Clone current repo
git clone https://github.com/LYH-YF/MWPToolkit.git && cd MWPToolkit

# Requirements
pip install -r requirements.txt
```

# Quick start

<span style="font-family: Open Sans; font-weight: 300; font-size: 16px; font-style: normal">300 Light normal hamburgefonstiv</span>

<span style="font-weight: 800; font-size: 20px">Evaluate a build-in dataset with a model</span>

<p style="color: blue">bar</p>

To have an initial try of our toolkit, you can use the provided script:

```
python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation --equation_fix=prefix --k_fold=5 --test_step=5 --gpu_id=0
```

Above script will run [GTS]() model on [Math23K]() dataset with 5 [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). It will take around **xx** minutes to train 5 GTS models independently and output the average scores of equation accuracy and value accuracy. The training log  can be found in the [log file](). 

If you would like to change the parameters, such as ```dataset``` and ```model```, please refer to the following instructions: **(YS: check whether my following description is correct)**

* ```model```: The model name you specify to apply. It should be chosen from options [].
* ```dataset```: The dataset name you specify to evaluate. It should be chosen from options [].
* ```task_type```: The type of generated equation. It should be chosen from options [single_equation | multi_equation]. Usually, it's up to the datasets **(YS: we can provide more details about this. Maybe for example, but we need to mention that how to know the correct choice for specific dataset)**.
* ```equation_fix```: The type of equation generation order. It should be chosen from options [infix | postfix | prefix]. Please note some models require specific type of equation generation order, so set this parameter to avoid bad performance because of the incorrect order type **(YS: can you give more details about how to avoid bad performance?)**.
* ```k_fold```: The fold number of cross-validation. It could be either NA value or interger. If it is NA value, it will run train-valid-test split procedure. 
* ```test_step```: The epoch number of training after which conducts the evaluation on test. It should be an interger.
* ```gpu_id```: The GPU ID for training the model. It should be an integer based on your GPU configuration. Please note that we haven't tested the framework with multipul GPUs yet.

**Evaluate a new dataset**

Our supported datasets are all saved under the folder ```'dataset'```. Besides trying our code with these build-in datasets, we also provide the option for you to run models on your own data copies, you can follow the steps below:

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

<strong>Finally</strong>:

Run:

```python run_mwptoolkit.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --gpu_id=0```

If you don't move your dataset folder and dataset configuration file to specific folder,these parameters can be set directly.

* ```dataset_path```, the default value is like ```dataset/dataset_name```, you can set your own dataset road path ```--dataset_path=[your_dataset]``` in cmd line.
* ```dataset_config_path```, the default value is like ```mwptoolkit/properties/dataset/dataset_name.json```, you can set your own dataset configuration file ```--dataset_config_path=[your_dataset_configuration]``` in cmd line.

**Example to run hyper-parameters search**

We implemented hyper-parameter search in our framework based ```ray.tune```.

You can run cmd line template below:

```python run_hyper_search.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --cpu_per_trial=2 --gpu_per_trial=0.5 --samples=1 --search_file=search_file.json --gpu_id=0```

* ```cpu_per_trial```, CPU resources to allocate per trial.
* ```gpu_per_trial```, GPU resources to allocate per trial.
* ```samples```, times to sample from the search space.
* ```search_file```, a json file including search parameter name and space.
* ```search_parameter```, if you don't write the search file, you can set this parameter to specify search space, e.g. ```--search_parameter=hidden_size=[256,512] --search_parameter=embedding_size=[64,128,256] --search_parameter=learning_rate='(1e-4, 1e-2)' ```

# Architecture

**Supoported Datasets**

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

**Models**

We have implemented 18 deep learning models and we are updating some other models. 

See two tables below.

**Implemented models** 

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
            <td align="center"><a href="https://aclanthology.org/D19-1241/">(Liu et al., 2019)</a></td>
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

**Updating models**
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

# Experiment Results

# Contributing

# Reference

# License

