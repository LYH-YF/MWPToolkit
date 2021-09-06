![](https://github.com/LYH-YF/MWPToolkit/blob/master/title.png)

##

[![PyPi Latest Release](https://img.shields.io/pypi/v/mwptoolkit)](https://pypi.org/project/mwptoolkit/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mwptoolkit/badge/?version=latest)](https://mwptoolkit.readthedocs.io/en/latest/?badge=latest)

[Doc](https://mwptoolkit.readthedocs.io/en/latest/)|[Model](#model)|[Dataset](#dataset)|[Paper](https://arxiv.org/pdf/2109.00799.pdf)

MWPToolkit is a PyTorch-based toolkit for Math Word Problem (MWP) solving. It is a comprehensive framework for research purpose that integrates popular MWP benchmark datasets and typical deep learning-based MWP algorithms. 

Our framework has the following architecture. You could utilize our toolkit to evaluate the build-in datasets, apply it to process your raw data copies or develop your own models. 

![](https://github.com/LYH-YF/MWPToolkit/blob/master/architecture1.png)
<div align="center"> Figure: The Overall Framework of MWP Toolkit </div>

## News

## Characteristics

* **Unification and Modularization**. We decouple solvers with different model architectures into highly modularized, reusable components and integrate them in a unified framework, which includes data, model, evaluation modules. It is convenient for you to study MWPs at a conceptual level and compare different models fairly.
* **Comprehensiveness and Standardization**. MWPToolkit has deployed the popular benchmark datasets and models for MWPs solving, covering Seq2Seq, Seq2Tree, Graph2Tree, and Pre-trained Language Models. Moreover, some tricks like hyper-parameter tuning used in general NLP tasks are integrated. As all models can be implemented with a same experimental configuration, the evaluation of different models is standardized.
* **Extensibility and Usability**. MWPToolkit provides user-friendly interfaces for various functions or modules. And the components in the pipeline architecture are modeled as exchangeable modules. You can try different combinations of modules via simply changing the configuration file or command line. You can also easily develop your own models by replacing or extending corresponding modules with your proposed ones.

## Installation
Development environment:
```
python >= 3.6.0
pytorch >= 1.5.0
pyltp >= 0.2.1 (optional)
```

### Method 1: Install from pip

```
pip install mwptoolkit
```
### Method 2: Install from source

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

Above script will run [GTS](https://www.ijcai.org/proceedings/2019/736) model on [Math23K](https://aclanthology.org/D17-1088/) dataset with 5 [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). It will take around **xx** minutes to train 5 GTS models independently and output the average scores of equation accuracy and value accuracy. The training log can be found in the [log file](https://github.com/LYH-YF/MWPToolkit/tree/master/log). 

If you would like to change the parameters, such as ```dataset``` and ```model```, please refer to the following instruction: 

* ```model```: The model name you specify to apply. You can see all options in Section [Model](#model).
* ```dataset```: The dataset name you specify to evaluate. You can see all options in Section [Dataset](#dataset).
* ```task_type```: The type of generated equation. It should be chosen from options [single_equation | multi_equation]. Usually, it's up to the datasets. You can refer to [dataset](#dataset). The single-equation dataset corresponds to 'single_equation' in code and multiple-equation dataset corresponds 'multi_equation' in code.
* ```equation_fix```: The type of equation generation order. It should be chosen from options [infix | postfix | prefix]. Please note some models require a specific type of equation generation order. Usually, the corresponding paper for model will mention which order it takes. You can look up the reference paper in Section [Model](#model).
* ```k_fold```: The fold number of cross-validation. It could be either NA value or an integer. If it is NA value, it will run train-valid-test split procedure. 
* ```test_step```: The epoch number of training after which conducts the evaluation on test. It should be an interger.
* ```gpu_id```: The GPU ID for training the model. It should be an integer based on your GPU configuration. Please note that we haven't tested the framework with multiple GPUs yet.

Please note ```model```, ```dataset``` and ```task_type``` are the required. We also provide the [interface](https://mwptoolkit.readthedocs.io/en/latest/_static/cmd.html) where you can config your experiments by clicking options and we automatically generate corresponding cmd lines. 

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

### Run hyper-parameters search

Our toolkit also provides the option to do hyper-parameters search, which could facilitate users to obtain optimal hyper-parameters efficiently. We integrated hyper-parameter search in our framework via ```ray.tune```. Due to the search procedure, it will take longer time to train a model.

You can run the cmd script template below:

```
python run_hyper_search.py --model=[model_name] --dataset=[dataset_name] --task_type=[single_equation|multi_equation] --equation_fix=[infix|postfix|prefix] --k_fold=[5|None] --cpu_per_trial=2 --gpu_per_trial=0.5 --samples=1 --search_file=search_file.json --gpu_id=0
```

* ```cpu_per_trial```: The CPU resources to allocate per trial.
* ```gpu_per_trial```: The GPU resources to allocate per trial.
* ```samples```: The number of sampling times from the search space.
* ```search_file```: A json file including search parameter name and space. For example:```["embedding_size=[64,128,256]","hidden_size=[256,512]","learning_rate=(1e-4, 1e-2)"]```
* ```search_parameter```: If you don't have the search file, you can set this parameter in command line to specify the search space. For example:```--search_parameter=hidden_size=[256,512] --search_parameter=embedding_size=[64,128,256] --search_parameter=learning_rate='(1e-4, 1e-2)```.

## Architecture

We have shown the overall architecture of our toolkit in the above [Figure](#news). The configuration is specified via command line, external config files and internal config dictionaries. Multiple processors and dataloaders are integrated to process different forms of data. Models and evaluators take charge of doing the training and evaluation. Therefore, input datasets will get prepared and trained based on the specified configuration. You can refer to [documentation]() for more information. 

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
            <td>RNN</td>
            <td align="center"><a href="https://arxiv.org/abs/1409.3215">(Sutskever et al., 2014)</a></td>
        </tr>
        <tr>
            <td>RNNVAE</td>
            <td align="center"><a href="https://arxiv.org/abs/1605.07869">(Zhang et al., 2016)</a></td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1706.03762">(Vaswani et al., 2017)</a></td>
        </tr>
        <tr>
            <td rowspan="5">Seq2Tree</td>
            <td>TRNN</td>
            <td align="center"><a href="https://ojs.aaai.org//index.php/AAAI/article/view/4697">(Wang et al., 2019)</a></td>
        </tr>
        <tr>
            <td>AST-Dec</td>
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
            <td>TSN</td>
            <td align="center"><a href="https://www.ijcai.org/proceedings/2020/555">(Zhang et al., 2020)</a></td>
        </tr>
        <tr>
            <td rowspan="2">Graph2Tree</td>
            <td>Graph2Tree</td>
            <td align="center"><a href="https://aclanthology.org/2020.acl-main.362/">(Zhang et al., 2020)</a></td>
        </tr>
        <tr>
            <td>MultiE&D</td>
            <td align="center"><a href="https://aclanthology.org/2020.coling-main.262/">(Shen et al., 2020)</a></td>
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
        <tr>
            <td rowspan="3">Updating</td>
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


### Evaluation metric

We have implemented 2 evaluation metrics to measure the effect of MWP models. 

<table align="center">
    <thread>
        <tr>
            <th align="center">measurement</th>
            <th align="center">note</th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td>Equ acc</td>
            <td align="center">The predicted equation is exactly match the correct equation
        </tr>
        <tr>
            <td>Val acc</td>
            <td align="center">The predicted answer is match the correct answer
        </tr>
    </tbody>
</table>

## Experiment Results

We have implemented the models on the datasets that are integrated within our toolkit. All the implementation follows the build-in configurations. All the experiments are conducted with 5 cross-validation. The experiment results(Equ acc|Val acc) are displayed in the following table.

### <div align="center">Single-equation Task Results</div>
<table align="center">
    <tr>
        <th rowspan="3">model</th>
        <td align="center" colspan="8"><strong>Dataset</strong></td>
    </tr>
    <tr>
        <th colspan="2">math23k</th>
        <th colspan="2">mawps-single</th>
        <th colspan="2">asdiv-a</th>
        <th colspan="2">mawps_asdiv-a_svamp</th>
    </tr>
    <tr>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
    </tr>
    <tr>
        <td>DNS</td>
        <td>57.1</td>
        <td>67.5</td>
        <td>78.9</td>
        <td>86.3</td>
        <td>63.0</td>
        <td>66.2</td>
        <td>22.1</td>
        <td>24.2</td>
    </tr>
    <tr>
        <td>MathEN</td>
        <td>66.7</td>
        <td>69.5</td>
        <td>85.9</td>
        <td>86.4</td>
        <td>64.3</td>
        <td>64.7</td>
        <td>21.8</td>
        <td>25.0</td>
    </tr>
    <tr>
        <td>Saligned</td>
        <td>59.1</td>
        <td>69.0</td>
        <td>86.0</td>
        <td>86.3</td>
        <td>66.0</td>
        <td>67.9</td>
        <td>23.9</td>
        <td>26.1</td>
    </tr>
    <tr>
        <td>GroupATT</td>
        <td>56.7</td>
        <td>66.6</td>
        <td>84.7</td>
        <td>85.3</td>
        <td>59.5</td>
        <td>61.0</td>
        <td>19.2</td>
        <td>21.5</td>
    </tr>
    <tr>
        <td>AttSeq</td>
        <td>57.1</td>
        <td>68.7</td>
        <td>79.4</td>
        <td>87.0</td>
        <td>64.2</td>
        <td>68.3</td>
        <td>23.0</td>
        <td>25.4</td>
    </tr>
    <tr>
        <td>LSTMVAE</td>
        <td>59.0</td>
        <td>70.0</td>
        <td>79.8</td>
        <td>88.2</td>
        <td>64.0</td>
        <td>68.7</td>
        <td>23.2</td>
        <td>25.9</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>52.3</td>
        <td>61.5</td>
        <td>77.9</td>
        <td>85.6</td>
        <td>57.2</td>
        <td>59.3</td>
        <td>18.4</td>
        <td>20.7</td>
    </tr>
    <tr>
        <td>TRNN</td>
        <td>65.0</td>
        <td>68.1</td>
        <td>86.0</td>
        <td>86.5</td>
        <td>68.9</td>
        <td>69.3</td>
        <td>22.6</td>
        <td>26.1</td>
    </tr>
    <tr>
        <td>AST-Dec</td>
        <td>57.5</td>
        <td>67.7</td>
        <td>84.1</td>
        <td>84.8</td>
        <td>54.5</td>
        <td>56.0</td>
        <td>21.9</td>
        <td>24.7</td>
    </tr>
    <tr>
        <td>GTS</td>
        <td>63.4</td>
        <td>74.2</td>
        <td>83.5</td>
        <td>84.1</td>
        <td>67.7</td>
        <td>69.9</td>
        <td>25.6</td>
        <td>29.1</td>
    </tr>
    <tr>
        <td>SAU-Solver</td>
        <td>64.6</td>
        <td>75.1</td>
        <td>83.4</td>
        <td>84.0</td>
        <td>68.5</td>
        <td>71.2</td>
        <td>27.1</td>
        <td>29.7</td>
    </tr>
    <tr>
        <td>TSN</td>
        <td>63.8</td>
        <td>74.4</td>
        <td>84.0</td>
        <td>84.7</td>
        <td>68.5</td>
        <td>71.0</td>
        <td>25.7</td>
        <td>29.0</td>
    </tr>
    <tr>
        <td>Graph2Tree</td>
        <td>64.9</td>
        <td>75.3</td>
        <td>84.9</td>
        <td>85.6</td>
        <td>72.4</td>
        <td>75.3</td>
        <td>31.6</td>
        <td>35.0</td>
    </tr>
    <tr>
        <td>MultiE&D</td>
        <td>65.5</td>
        <td>76.5</td>
        <td>83.2</td>
        <td>84.1</td>
        <td>70.5</td>
        <td>72.6</td>
        <td>29.3</td>
        <td>32.4</td>
    </tr>
    <tr>
        <td>BERTGen</td>
        <td>64.8</td>
        <td>76.6</td>
        <td>79.0</td>
        <td>86.9</td>
        <td>68.7</td>
        <td>71.5</td>
        <td>22.2</td>
        <td>24.8</td>
    </tr>
    <tr>
        <td>RoBERTaGen</td>
        <td>65.2</td>
        <td>76.9</td>
        <td>80.8</td>
        <td>88.4</td>
        <td>68.7</td>
        <td>72.1</td>
        <td>27.9</td>
        <td>30.3</td>
    </tr>
    <tr>
        <td>GPT-2</td>
        <td>63.8</td>
        <td>74.3</td>
        <td>75.4</td>
        <td>75.9</td>
        <td>59.9</td>
        <td>61.4</td>
        <td>22.5</td>
        <td>25.7</td>
    </tr>
</table>

### <div align="center">Multiple-equation Task Result</div>
<table align="center">
    <tr>
        <th rowspan="3">model</th>
        <td align="center" colspan="6"><strong>Dataset</strong></td>
    </tr>
    <tr>
        <th colspan="2">draw</th>
        <th colspan="2">hmwp</th>
    </tr>
    <tr>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
        <th>Equ. Acc</th>
        <th>Ans. Acc</th>
    </tr>
    <tr>
        <td>DNS</td>
        <td>35.8</td>
        <td>36.8</td>
        <td>24.0</td>
        <td>32.7</td>
    </tr>
    <tr>
        <td>MathEN</td>
        <td>38.2</td>
        <td>39.5</td>
        <td>32.4</td>
        <td>43.7</td>
    </tr>
    <tr>
        <td>Saligned</td>
        <td>36.7</td>
        <td>37.8</td>
        <td>31.0</td>
        <td>41.8</td>
    </tr>
    <tr>
        <td>GroupATT</td>
        <td>30.4</td>
        <td>31.4</td>
        <td>25.2</td>
        <td>33.2</td>
    </tr>
    <tr>
        <td>AttSeq</td>
        <td>39.7</td>
        <td>41.2</td>
        <td>32.9</td>
        <td>44.7</td>
    </tr>
    <tr>
        <td>LSTMVAE</td>
        <td>40.9</td>
        <td>42.3</td>
        <td>33.6</td>
        <td>45.9</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>27.1</td>
        <td>28.3</td>
        <td>24.4</td>
        <td>32.4</td>
    </tr>
    <tr>
        <td>TRNN</td>
        <td>27.4</td>
        <td>28.9</td>
        <td>27.2</td>
        <td>36.8</td>
    </tr>
    <tr>
        <td>AST-Dec</td>
        <td>26.0</td>
        <td>26.7</td>
        <td>24.9</td>
        <td>32.0</td>
    </tr>
    <tr>
        <td>GTS</td>
        <td>38.6</td>
        <td>39.9</td>
        <td>33.7</td>
        <td>44.6</td>
    </tr>
    <tr>
        <td>SAU-Solver</td>
        <td>38.4</td>
        <td>39.2</td>
        <td>33.1</td>
        <td>43.7</td>
    </tr>
    <tr>
        <td>TSN</td>
        <td>39.3</td>
        <td>40.4</td>
        <td>34.3</td>
        <td>44.9</td>
    </tr>
    <tr>
        <td>Graph2Tree</td>
        <td>39.8</td>
        <td>41.0</td>
        <td>34.4</td>
        <td>45.1</td>
    </tr>
    <tr>
        <td>MultiE&D</td>
        <td>38.1</td>
        <td>39.2</td>
        <td>34.6</td>
        <td>45.3</td>
    </tr>
    <tr>
        <td>BERTGen</td>
        <td>33.9</td>
        <td>35.0</td>
        <td>29.2</td>
        <td>39.5</td>
    </tr>
    <tr>
        <td>RoBERTaGen</td>
        <td>34.2</td>
        <td>34.9</td>
        <td>30.6</td>
        <td>41.0</td>
    </tr>
    <tr>
        <td>GPT-2</td>
        <td>30.7</td>
        <td>31.5</td>
        <td>36.3</td>
        <td>49.0</td>
    </tr>
</table>
## Contributing

We will keep updating and maintaining this repository. You are welcome to contribute to this repository through giving us suggestions and developing extensions! If you have any questions or encounter a bug, please fill an [issue](https://github.com/LYH-YF/MWPToolkit/issues). 

## Cite

If you find MWP toolkit is useful for your research, please cite:

```
@article{lan2021mwptoolkit,
    title={MWPToolkit: An Open-Source Framework for Deep Learning-Based Math Word Problem Solvers},
    author={Yihuai Lan and Lei Wang and Qiyuan Zhang and Yunshi Lan and Bing Tian Dai and Yan Wang and Dongxiang Zhang and Ee-Peng Lim},
    journal={arXiv preprint arXiv:2109.00799},
    year={2021}
}
```


## License

MWPToolkit uses [MIT License](https://github.com/LYH-YF/MWPToolkit/blob/master/LICENSE).
