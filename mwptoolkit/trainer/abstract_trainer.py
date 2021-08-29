# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:13:14
# @File: abstract_trainer.py


from logging import getLogger

import torch

from mwptoolkit.utils.utils import write_json_data

class AbstractTrainer(object):
    """abstract trainer

    the base class of trainer class.
    
    example of instantiation:
        
        >>> trainer = AbstractTrainer(config, model, dataloader, evaluator)

        for training:
            
            >>> trainer.fit()
        
        for testing:
            
            >>> trainer.test()
        
        for parameter searching:

            >>> trainer.param_search()
    """
    def __init__(self, config, model, dataloader, evaluator):
        """
        Args:
            config (config): An instance object of Config, used to record parameter information.
            model (Model): An object of deep-learning model. 
            dataloader (Dataloader): dataloader object.
            evaluator (Evaluator): evaluator object.
        
        expected that config includes these parameters below:

        test_step (int): the epoch number of training after which conducts the evaluation on test.

        best_folds_accuracy (list|None): when running k-fold cross validation, this keeps the accuracy of folds that already run. 

        """
        super().__init__()
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = getLogger()
        self.best_folds_accuracy=config["best_folds_accuracy"]
        self.test_step=config["test_step"]

        self.best_valid_equ_accuracy = 0.
        self.best_valid_value_accuracy = 0.
        self.best_test_equ_accuracy = 0.
        self.best_test_value_accuracy = 0.
        self.start_epoch = 0
        self.epoch_i = 0
        self.output_result=[]

    def _save_checkpoint(self):
        raise NotImplementedError

    def _load_checkpoint(self):
        raise NotImplementedError

    def _save_model(self):
        state_dict = {"model": self.model.state_dict()}
        if self.config["k_fold"]:
            path=self.config["trained_model_path"][:-4]+"-fold{}.pth".format(self.config["fold_t"])
            torch.save(state_dict,path)
        else:
            torch.save(state_dict, self.config["trained_model_path"])

    def _load_model(self):
        if self.config["k_fold"]:
            path=self.config["trained_model_path"][:-4]+"-fold{}.pth".format(self.config["fold_t"])
            state_dict = torch.load(path, map_location=self.config["map_location"])
        else:
            state_dict = torch.load(self.config["trained_model_path"], map_location=self.config["map_location"])
        #self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict["model"],False)
    
    def _save_output(self):
        if self.config['output_path']:
            if self.config["k_fold"]:
                path=self.config["output_path"][:-5]+"-fold{}.json".format(self.config["fold_t"])
                write_json_data(self.output_result,path)
            else:
                write_json_data(self.output_result,self.config['output_path'])

    def _build_optimizer(self):
        raise NotImplementedError

    def _train_batch(self):
        raise NotADirectoryError

    def _eval_batch(self):
        raise NotImplementedError

    def _train_epoch(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def evaluate(self, eval_set):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def param_search(self):
        raise NotImplementedError