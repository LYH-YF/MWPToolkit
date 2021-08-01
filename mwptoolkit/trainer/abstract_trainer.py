from logging import getLogger

import torch

from mwptoolkit.utils.utils import write_json_data

class AbstractTrainer(object):
    def __init__(self, config, model, dataloader, evaluator):
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
