import sys
import os
from logging import getLogger
from enum import Enum

import torch

from mwptoolkit.utils.utils import read_json_data, get_model


class Config(object):
    """The class for loading pre-defined parameters.

    Config will load the parameters internal config file, dataset config file, model config file, config dictionary and cmd line.

    The default road path of internal config file is 'mwptoolkit/config/config.json', and it's not supported to change.

    The dataset config, model config and config dictionary are called the external config. 
    
    According to specific dataset and model, this class will load the dataset config from default road path 'mwptoolkit/properties/dataset/dataset_name.json' 
    and model config from default road path 'mwptoolkit/properties/model/model_name.json'.

    You can set the parameters 'model_config_path' and 'dataset_config_path' to load your own model and dataset config, but note that only json file can be loaded correctly.
    Config dictionary is a dict-like object. When you initialize the Config object, you can pass config dictionary through the code 'config = Config(config_dict=config_dict)'

    Cmd line requires you keep the template --param_name=param_value to set any parameter you want.

    If there are multiple values of the same parameter, the priority order is as following:

    cmd line > external config > internal config
    
    in external config, config dictionary > model config > dataset config.

    """
    def __init__(self, model_name=None, dataset_name=None, task_type=None, config_dict={}):
        """
        Args:
            model (str): the model name, default is None, if it is None, config will search the parameter 'model'
            from the external input as the model name.
            
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            
            task_type (str): the task type, default is None, if it is None, config will search the parameter 'task_type'
            from the external input as the task type.
            
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        super().__init__()
        self.file_config_dict = {}
        self.cmd_config_dict = {}
        self.config_dict = {}
        self.external_config_dict = {}
        self.model_config_dict = {}
        self.dataset_config_dict = {}
        self.path_config_dict = {}  #
        self.final_config_dict = {}
        self._load_config()
        self._merge_config_dict(model_name, dataset_name, task_type, config_dict)
        self._load_cmd_line()

        self._build_path_config()  #
        self._load_model_config()  #
        self._load_dataset_config()  #

        self._merge_external_config_dict()
        self._build_final_config_dict()

        self._init_model_path()
        self._init_device()

    def _load_config(self):
        self.file_config_dict = read_json_data('mwptoolkit/config/config.json')

    def _merge_config_dict(self, model_name, dataset_name, task_type, config_dict):
        self.config_dict['model'] = model_name
        self.config_dict['dataset'] = dataset_name
        self.config_dict['task_type'] = task_type
        self.config_dict.update(config_dict)
        

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum, None)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    elif param.lower() == "none":
                        value = None
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    if arg.startswith("--search_parameter"):
                        continue
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in Mwptoolkit'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)

        if 'task_type' not in cmd_config_dict:
            task_type=self.config_dict['task_type']
        else:
            task_type=cmd_config_dict['task_type']
        if task_type not in ['single_equation', 'multi_equation']:
            raise NotImplementedError("task_type {} can't be found".format(task_type))
        self.cmd_config_dict.update(cmd_config_dict)
        
        for key, value in self.config_dict.items():
            try:
                self.config_dict[key] = self.cmd_config_dict[key]
            except:
                pass
        for key, value in self.file_config_dict.items():
            try:
                self.file_config_dict[key] = self.cmd_config_dict[key]
            except:
                pass
        return cmd_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError('model need to be specified in at least one of the these ways: ' '[model variable, config file, config dict, command line] ')
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError('dataset need to be specified in at least one of the these ways: ' '[dataset variable, config file, config dict, command line] ')
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _load_model_config(self):
        if self.file_config_dict["load_best_config"]:
            model_config_path=self.path_config_dict["best_config_path"]
        else:
            model_config_path=self.path_config_dict["model_config_path"]
        try:
            self.model_config_dict = read_json_data(model_config_path)
        except:
            self.model_config_dict = {}
        for key, value in self.model_config_dict.items():
            try:
                self.model_config_dict[key] = self.config_dict[key]
            except:
                pass
            try:
                self.model_config_dict[key] = self.cmd_config_dict[key]
            except:
                pass

    def _load_dataset_config(self):
        try:
            self.dataset_config_dict = read_json_data(self.path_config_dict["dataset_config_path"])
        except:
            self.dataset_config_dict = {}
        for key, value in self.dataset_config_dict.items():
            try:
                self.dataset_config_dict[key] = self.config_dict[key]
            except:
                pass
            try:
                self.dataset_config_dict[key] = self.cmd_config_dict[key]
            except:
                pass

    def _build_path_config(self):
        path_config_dict = {}
        root=os.getcwd()
        path_config_dict['root']=root
        model_name = self.config_dict['model']
        dataset_name = self.config_dict['dataset']
        if model_name == None:
            model_name = self.cmd_config_dict["model"]
        if dataset_name == None:
            dataset_name = self.cmd_config_dict["dataset"]
        path_config_dict["model_config_path"] = "mwptoolkit/properties/model/{}.json".format(model_name)
        path_config_dict["best_config_path"] = "mwptoolkit/properties/best_config/{}_{}.json".format(model_name,dataset_name)
        path_config_dict["dataset_config_path"] = "mwptoolkit/properties/dataset/{}.json".format(dataset_name)
        path_config_dict["dataset_path"] = "dataset/{}".format(dataset_name)
        self.path_config_dict = path_config_dict

        for key, value in path_config_dict.items():
            try:
                self.path_config_dict[key] = self.config_dict[key]
            except:
                pass
            try:
                self.path_config_dict[key] = self.cmd_config_dict[key]
            except:
                pass

    def _init_model_path(self):
        path_config_dict = {}
        model_name = self.final_config_dict["model"]
        dataset_name = self.final_config_dict["dataset"]
        fix = self.final_config_dict["equation_fix"]
        path_config_dict["checkpoint_path"] = 'checkpoint/' + '{}-{}-{}.pth'.format(model_name, dataset_name, fix)
        path_config_dict["trained_model_path"] = 'trained_model/' + '{}-{}-{}.pth'.format(model_name, dataset_name, fix)
        path_config_dict["log_path"] = 'log/' + '{}-{}-{}.log'.format(model_name, dataset_name, fix)
        for key, value in path_config_dict.items():
            try:
                path_config_dict[key] = self.cmd_config_dict[key]
            except:
                pass
        self.path_config_dict.update(path_config_dict)
        self.final_config_dict.update(path_config_dict)

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.path_config_dict)
        external_config_dict.update(self.dataset_config_dict)
        external_config_dict.update(self.model_config_dict)
        external_config_dict.update(self.config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _build_final_config_dict(self):
        self.final_config_dict.update(self.file_config_dict)
        self.final_config_dict.update(self.external_config_dict)

    def _init_device(self):
        if self.final_config_dict["gpu_id"] == None:
            if torch.cuda.is_available() and self.final_config_dict["use_gpu"]:
                self.final_config_dict["gpu_id"] = "0"
            else:
                self.final_config_dict["gpu_id"] = ""
        else:
            if self.final_config_dict["use_gpu"] != True:
                self.final_config_dict["gpu_id"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict["gpu_id"])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.final_config_dict["map_location"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.final_config_dict['gpu_nums'] = torch.cuda.device_count()

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __str__(self):
        args_info = ''
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info
