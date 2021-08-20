# <div align="center"> MWPToolkit </div>

![](https://img.shields.io/badge/license-MIT-green)

[Doc]()|[Model](#model)|[Dataset](#dataset)|[Paper]()

MWPToolkit is a PyTorch-based toolkit for Math Word Problem(MWP) solving. It is a comprehensive framework for research purpose that integrates popular MWP benchmark datasets and typical deep learning-based MWP algorithms. 

Our framework has the following architecture. You could utilize our toolkit to evaluate the build-in datasets, apply it to process your raw data copies or develop your own models. **(YS: draw a figure about the framework architecture)**

![](https://octodex.github.com/images/yaktocat.png)
<div align="center"> Figure: Architecture of MWP Toolkit </div>

## News

## Feature

**(YS: can you come up with more points?)**

* **Comprehensive toolkit for MWP solving task**. To our best knowledge, MWP toolkit is the first open-source library for MWP solving task, where popular benchmark datasets and advanced deep learning-based methods for MWP solving tasks are integrated into a unified framework. 
* **Easy to get started**. MWP toolkit is developed upon Python and Pytorch. We provide detailed instruction, which facilitates users to evaluate the build-in datasets or apply the code to their own data.
* **Highly modularized framework**. MWP toolkit is designed with highly reused modules and provides convenient interfaces for users. Specifically, data preprocessor, data loader, encoder, decoder and evaluator form the running procedure. Each module could be developed and extended independently.


## Installation
Development environment **(YS: move other libraries expect python, pytorch to requirements.txt file)**:
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

### Method 1: Install from pip

**(YS: figure out how to make pip install of this library)**
```
pip install 
```
### Method 2: Install from source

**(YS: check if this is correct)**
```
# Clone current repo
git clone https://github.com/LYH-YF/MWPToolkit.git && cd MWPToolkit

# Requirements
pip install -r requirements.txt
```

## Quick Start

### Evaluate a build-in dataset with a model

To have an initial trial of our toolkit, you can use the provided cmd script:

```
python run_mwptoolkit.py --model=GTS --dataset=math23k --task_type=single_equation --equation_fix=prefix --k_fold=5 --test_step=5 --gpu_id=0
```

Above script will run [GTS](https://www.ijcai.org/proceedings/2019/736) model on [Math23K](https://aclanthology.org/D17-1088/) dataset with 5 [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). It will take around **xx** minutes to train 5 GTS models independently and output the average scores of equation accuracy and value accuracy. The training log can be found in the [log file](). 

If you would like to change the parameters, such as ```dataset``` and ```model```, please refer to the following instruction: **(YS: check whether my following description is correct, maybe we just keep the required arguments)**

* ```model```*: The model name you specify to apply. You can see all options in Section [Model](#model).
* ```dataset```*: The dataset name you specify to evaluate. You can see all options in Section [Dataset](#dataset).
* ```task_type```*: The type of generated equation. It should be chosen from options [single_equation | multi_equation]. Usually, it's up to the datasets **(YS: we can provide more details about this. Maybe for example, but we need to mention that how to know the correct choice for specific dataset)**.
* ```equation_fix```: The type of equation generation order. It should be chosen from options [infix | postfix | prefix]. Please note some models require a specific type of equation generation order, so set this parameter to avoid bad performance because of the incorrect order type **(YS: can you give more details about how to avoid bad performance?)**.
* ```k_fold```: The fold number of cross-validation. It could be either NA value or an integer. If it is NA value, it will run train-valid-test split procedure. 
* ```test_step```: The epoch number of training after which conducts the evaluation on test. It should be an interger.
* ```gpu_id```: The GPU ID for training the model. It should be an integer based on your GPU configuration. Please note that we haven't tested the framework with multiple GPUs yet.

Please note ```model```, ```dataset``` and ```task_type``` are the required. 

### Evaluate a new dataset

Our supported datasets are all saved under the folder ```'dataset'```. Besides trying our code with these build-in datasets, we also provide the option for you to run models on your own data copies, you can follow the steps below:

<strong>Step 1</strong>: Organize your dataset. Your dataset folder (same as the dataset name) should include three json files for train, validation and test, respectively: 

```
dataset_name
    |----trainset.json
    |----validset.json
    |----testset.json
```

Move your dataset folder under path ```'dataset'``` of our framework, the file structure would be like:

```
dataset
    |----dataset_name
            |----trainset.json
            |----validset.json
            |----testset.json
```

<strong>Step 2</strong>: Setup your dataset configuration. The dataset configuration files are saved under path ```'mwptoolkit/properties/dataset/'```. You can write your own dataset configuration and save a JSON file under the path. The path to your JSON file should be ```mwptoolkit/properties/dataset/dataset_name.json```

<strong>Step 3</strong>: Run the code!

```
python run_mwptoolkit.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --gpu_id=0
```

Instead of moving your dataset folder and dataset configuration file to the above folders, the following parameters can be set directly.

* ```dataset_path```: The path to dataset folder. The default value is ```'dataset/dataset_name'```, you can change it to your own dataset path via appending ```--dataset_path=[your_dataset]``` to cmd script.
* ```dataset_config_path```: The path to dataset configuration file. The default value is ```'mwptoolkit/properties/dataset/dataset_name.json'```, you can change it to your own dataset configuration path via appending ```--dataset_config_path=[your_dataset_configuration]``` to cmd script.

### Run Hyper-parameters Search

Our toolkit also provides the option to do hyper-parameters search, which could facilitate users to obtain optimal hyper-parameters efficiently. We integrated hyper-parameter search in our framework via ```ray.tune```.

You can run the cmd script template below:

```
python run_hyper_search.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --cpu_per_trial=2 --gpu_per_trial=0.5 --samples=1 --search_file=search_file.json --gpu_id=0
```

* ```cpu_per_trial```: The CPU resources to allocate per trial.
* ```gpu_per_trial```: The GPU resources to allocate per trial.
* ```samples```: The number of sampling times from the search space.
* ```search_file```: A json file including search parameter name and space**(YS: maybe we provide a code sample)**.
* ```search_parameter```: If you don't have the search file, you can set this parameter to specify the search space. For example, ```--search_parameter=hidden_size=[256,512]```, ```--search_parameter=embedding_size=[64,128,256]``` and ```--search_parameter=learning_rate='(1e-4, 1e-2)``` **(YS: what do you mean here?)**.

## Architecture

We have shown the overall architecture of our toolkit in the above figure. **(YS: to complete later)**

### Dataset

We have deployed **8** popular MWP datasets in our toolkit. These datasets are divided into two categories, **Single-equation** dataset and **Multiple-equation** dataset, which can be found in the table below. We will keep updating more datasets like ape200k<a href="https://arxiv.org/abs/2009.11506">(Zhao et al., 2020)</a>, dolphin1878<a href="https://aclanthology.org/D15-1135/">(Shi et al., 2015)</a> and dolphin18k<a href="https://aclanthology.org/P16-1084/">(Huang et al., 2016)</a>. 

<table align="center">
    <thread align="center">
        <tr>
            <th align="center">task</th>
            <th align="center">dataset</th>
            <th align="center">reference</th>
        </tr>
    </thread>
    <tbody align="center">
        <tr>
            <td rowspan="4" align="center">Single-equation dataset</td>
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
            <td rowspan="4" align="center">Multiple-equation dataset</td>
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

### Model

We have deployed **18** deep learning MWP models in our toolkit. Based on the featured generation procedure, we categorize them into **Sequence-to-sequence**, **Sequence-to-tree**, **Graph-to-tree**, **VAE** and **Pre-trained** models. Please note Pre-trained models are simple implementation of pretrained language models on MWP solving task. The table is displayed as follows:

<table align="center">
    <thread>
        <tr>
            <th align="center">type</th>
            <th align="center">model</th>
            <th align="center">reference</th>
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
            <td rowspan="3">Pre-trained</td>
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

### Updating models

**(YS: can you merge the following table to above one?)**

<table align="center">
    <thread>
        <tr>
            <th align="center">model</th>
            <th align="center">reference</th>
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

### Evaluator

We have implemented two evaluators to measure the effect of MWP models. 

<table align="center">
    <thread>
        <tr>
            <th align="center">evaluator</th>
            <th align="center">note</th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td>Equ accuracy</td>
            <td align="center">The predicted equation is exactly match the correct equation
        </tr>
        <tr>
            <td>Val accuracy</td>
            <td align="center">The predicted answer is match the correct answer
        </tr>
    </tbody>
</table>

## Experiment Results

## Contributing

We will keep updating and maintaining this repository. You are welcome to contribute to this repository through giving us suggestions and developing extensions! If you have any questions or encounter a bug, please fill an [issue](https://github.com/LYH-YF/MWPToolkit/issues). 

## Cite

If you find MWP toolkit is useful for your research, please cite:

> @article{
> ...
> }

## License

