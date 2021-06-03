import torch
import time
from logging import getLogger
from torch import nn
import numpy as np
from mwptoolkit.utils.utils import time_since
from mwptoolkit.utils.enum_type import PAD_TOKEN, DatasetType,TaskType,SpecialTokens
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.loss.binary_cross_entropy_loss import BinaryCrossEntropyLoss
from mwptoolkit.module.Optimizer.optim import WarmUpScheduler
from mwptoolkit.loss.mse_loss import MSELoss


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


class Trainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _build_loss(self, symbol_size, out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight, pad)

    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def _train_batch(self, batch):
        outputs = self.model(batch["question"], batch["ques len"], batch["equation"])
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
            self.loss.eval_batch(outputs, batch_equation.view(-1))
        else:
            self.loss.eval_batch(outputs, batch["equation"].view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"], batch["ques len"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()

            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                self.logger.info("----------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                self.logger.info("----------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                if valid_val_ac >= self.best_valid_value_accuracy:
                    self.best_valid_value_accuracy = valid_val_ac
                    self.best_valid_equ_accuracy = valid_equ_ac
                    self.best_test_value_accuracy = test_val_ac
                    self.best_test_equ_accuracy = test_equ_ac
                    self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))


class SingleEquationTrainer(Trainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _build_loss(self, symbol_size, out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight, pad)

    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def _train_batch(self, batch):
        outputs = self.model(batch["question"], batch["ques len"], batch["equation"])
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
            self.loss.eval_batch(outputs, batch_equation.view(-1))
        else:
            self.loss.eval_batch(outputs, batch["equation"].view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"], batch["ques len"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            #val_ac, equ_ac, _, _ = self.evaluator.result(target[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
            
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

class MultiEquationTrainer(Trainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _build_loss(self, symbol_size, out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight, pad)

    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def _train_batch(self, batch):
        outputs = self.model(batch["question"], batch["ques len"], batch["equation"])
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
            self.loss.eval_batch(outputs, batch_equation.view(-1))
        else:
            self.loss.eval_batch(outputs, batch["equation"].view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"], batch["ques len"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            #val_ac, equ_ac, _, _ = self.evaluator.result_multi(target[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()

            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))


class GTSTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self.loss = MaskedCrossEntropyLoss()

    def _build_optimizer(self):
        # optimizer
        self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.decoder_optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.node_generater_optimizer = torch.optim.Adam(self.model.node_generater.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.merge_optimizer = torch.optim.Adam(self.model.merge.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # scheduler
        self.embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.node_generater_scheduler = torch.optim.lr_scheduler.StepLR(self.node_generater_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(self.merge_optimizer, step_size=self.config["step_size"], gamma=0.5)

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "embedder_optimizer": self.embedder_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_optimizer": self.node_generater_optimizer.state_dict(),
            "merge_optimizer": self.merge_optimizer.state_dict(),
            "embedder_scheduler": self.embedder_scheduler.state_dict(),
            "encoder_scheduler": self.encoder_scheduler.state_dict(),
            "decoder_scheduler": self.decoder_scheduler.state_dict(),
            "generate_scheduler": self.node_generater_scheduler.state_dict(),
            "merge_scheduler": self.merge_scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.embedder_optimizer.load_state_dict(check_pnt["embedder_optimizer"])
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.decoder_optimizer.load_state_dict(check_pnt["decoder_optimizer"])
        self.node_generater_optimizer.load_state_dict(check_pnt["generate_optimizer"])
        self.merge_optimizer.load_state_dict(check_pnt["merge_optimizer"])
        #load parameter of scheduler
        self.embedder_scheduler.load_state_dict(check_pnt["embedder_scheduler"])
        self.encoder_scheduler.load_state_dict(check_pnt["encoder_scheduler"])
        self.decoder_scheduler.load_state_dict(check_pnt["decoder_scheduler"])
        self.node_generater_scheduler.load_state_dict(check_pnt["generate_scheduler"])
        self.merge_scheduler.load_state_dict(check_pnt["merge_scheduler"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _scheduler_step(self):
        self.embedder_scheduler.step()
        self.encoder_scheduler.step()
        self.decoder_scheduler.step()
        self.node_generater_scheduler.step()
        self.merge_scheduler.step()

    def _optimizer_step(self):
        self.embedder_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.node_generater_optimizer.step()
        self.merge_optimizer.step()

    def _model_zero_grad(self):
        self.model.embedder.zero_grad()
        self.model.encoder.zero_grad()
        self.model.decoder.zero_grad()
        self.model.node_generater.zero_grad()
        self.model.merge.zero_grad()

    def _model_train(self):
        self.model.embedder.train()
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.node_generater.train()
        self.model.merge.train()

    def _model_eval(self):
        self.model.embedder.eval()
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.node_generater.eval()
        self.model.merge.eval()

    def _train_batch(self, batch):
        '''
        seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                UNK_TOKEN,num_start,target=None, target_length=None,max_length=30,beam_size=5
        '''
        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]

        outputs, _=self.model(batch["question"],batch["ques len"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start,batch["equation"],batch["equ len"],UNK_TOKEN=unk)
        self.loss.eval_batch(outputs, batch["equation"], batch["equ mask"])
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        # if batch["id"]==[80313]:
        #     print(1)
        # else:
        #     return [False], [False]
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]
        test_out=self.model(batch["question"],batch["ques len"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start)

        if self.config["task_type"]==TaskType.SingleEquation:
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out, batch["equation"].tolist()[0], batch["num list"][0], batch["num stack"][0])
        elif self.config["task_type"]==TaskType.MultiEquation:
            val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out, batch["equation"].tolist()[0], batch["num list"][0], batch["num stack"][0])
        else:
            raise NotImplementedError
        return [val_ac], [equ_ac]

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self._model_train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self._model_zero_grad()
            
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self._optimizer_step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost
        #print("epoch [%2d]avr loss [%2.8f]"%(self.epoch_i,loss_total /self.batch_nums))
        #print("epoch train time {}".format(time_since(time.time() -epoch_start_time)))

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]
        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self._scheduler_step()
            
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
        

    def evaluate(self, eval_set):
        self._model_eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

class GTSWeakTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self.loss = MaskedCrossEntropyLoss()

    def _build_buffer_batch(self):
        self._buffer_batches = [[] for i in range(self.dataloader.trainset_nums)]
        self._buffer_batches_exp = [[] for i in range(self.dataloader.trainset_nums)]

    def _train_batch(self, batch):
        '''
        seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                UNK_TOKEN,num_start,target=None, target_length=None,max_length=30,beam_size=5
        '''
        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]

        outputs=self.model(batch["question"],batch["ques len"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start,batch["equation"],batch["equ len"],UNK_TOKEN=unk)
        self.loss.eval_batch(outputs, batch["equation"], batch["equ mask"])
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _train_epoch(self):
        epoch_start_time = time.time() #
        loss_total = 0.  #
        #self._model_train()
        self.mask_flag = False #
        self._pos = 0 #
        self.epo_iteration = 0
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self._model_train()
            self._model_zero_grad()
            if self.epoch_i == 1 and self.batch_idx <= 2: #
                self.mask_flag = True
            buffer_batches_train = self._buffer_batches[self._pos: self._pos + len(batch["ques len"])]
            buffer_batches_train_exp = self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])]
            iterations, buffer_batch_new, buffer_batch_exp, batch_loss = self._weakly_train_batch(batch, buffer_batches_train, buffer_batches_train_exp)
            loss_total += batch_loss
            self.epo_iteration += iterations
            self._buffer_batches[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_new
            self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_exp
            self._pos += len(batch["ques len"])
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        loss_total = loss_total if self.epo_iteration == 0 else loss_total/self.epo_iteration
        return loss_total, epoch_time_cost

    def _weakly_train_batch(self, batch, buffer_batches_train, buffer_batches_train_exp):
        batch_size = len(batch["ques len"])

        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in
                         self.dataloader.dataset.generate_list]
        input_var = batch["question"]  # [batch_size, max_len]

        input_var = input_var.transpose(0, 1)
        fix_input_length, fix_target_list, \
        fix_index, fix_target_length, \
        buffer_batch_new, buffer_batch_exp = self.model.weakly_train(batch["question"], batch["ques len"], batch['ans'],
                                                                     batch["num list"], batch["num size"], \
                                                                     generate_nums, batch["num pos"], num_start,
                                                                     batch["equation"], batch["equ len"],
                                                                     self.epoch_i - 1, self.mask_flag, unk, \
                                                                     self.config['supervising_mode'],
                                                                     buffer_batches_train, buffer_batches_train_exp,
                                                                     Lang=self.dataloader, n_step=50)
        # buffer_batch_new, buffer_batch_exp = Weakly_Supervising(generate_exps_list, all_node_outputs_mask_list,buffer_batches_train, buffer_batches_train_exp)

        # print("fix_input_length", fix_input_length)
        # print("fix_target_list", fix_target_list)

        fix_input_length = np.array(fix_input_length)
        fix_target_list = np.array(fix_target_list)
        fix_index = np.array(fix_index)
        fix_target_length = np.array(fix_target_length)

        inds = np.argsort(-fix_input_length)
        # print("inds", inds)
        fix_target_list = fix_target_list[inds].tolist()
        # print("fix_target_list", fix_target_list)
        fix_index = fix_index[inds].tolist()
        # print("fix_index", fix_index)
        fix_target_length = fix_target_length[inds].tolist()
        # print("fix_target_length", fix_target_length)
        # print(batch_size)
        mapo_batch_size = 64
        if not len(fix_target_list) % mapo_batch_size == 0:
            num_iteration = int(len(fix_target_list) / mapo_batch_size) + 1
        else:
            num_iteration = int(len(fix_target_list) / mapo_batch_size)
        # print("num_iteration", num_iteration)
        batch_loss = torch.FloatTensor([[0.0]]).to(self.config["device"])
        for j in range(num_iteration):
            if not j * mapo_batch_size + mapo_batch_size - 1 < len(fix_target_list):
                mapo_batch_size = len(fix_target_list) - ((j - 1) * mapo_batch_size + mapo_batch_size)
            target_length_mapo = fix_target_length[j * mapo_batch_size: (j * mapo_batch_size + mapo_batch_size)]
            idx_list = fix_index[j * mapo_batch_size: (j * mapo_batch_size + mapo_batch_size)]
            target_list = fix_target_list[j * mapo_batch_size: (j * mapo_batch_size + mapo_batch_size)]
            input_length_mapo = []
            num_pos_mapo = []
            num_size_mapo_batch = []
            # print("input_length", batch["ques len"])

            for k in range(mapo_batch_size):
                idx = idx_list[k]
                input_length_mapo.append(batch["ques len"][idx])
                num_pos_mapo.append(batch["num pos"][idx])
                num_size_mapo_batch.append(batch["num size"][idx])
            # print("input_length_mapo", input_length_mapo)

            input_var_mapo = torch.zeros((max(input_length_mapo), mapo_batch_size), dtype=torch.long)
            # print("input_var_mapo", input_var_mapo.size())
            target = torch.zeros((mapo_batch_size, max(target_length_mapo)), dtype=torch.long).to(self.config["device"])

            for k in range(mapo_batch_size):
                idx = idx_list[k]
                input_var_mapo[:, k] = input_var[:, idx][:max(input_length_mapo)]
                target[k][:target_length_mapo[k]] = torch.LongTensor(target_list[k])

            # print("target", target.size())
            self._model_train()
            self._model_zero_grad()
            input_var_mapo = input_var_mapo.to(self.config["device"])
            input_var_mapo = input_var_mapo.transpose(0, 1)

            output, target = self.model(input_var_mapo, input_length_mapo, batch["num stack"], num_size_mapo_batch,
                                generate_nums, num_pos_mapo, \
                                num_start, target=target, target_length=target_length_mapo, UNK_TOKEN=unk)
            target_length_mapo = torch.LongTensor(target_length_mapo).to(self.config["device"])
            target_mask_mapo = self.sequence_mask(target_length_mapo)
            self.loss.eval_batch(output, target, target_mask_mapo)
            batch_loss += self.loss.get_loss()
            self.loss.backward()
            self._optimizer_step()
            self.loss.reset()
        # self._buffer_batches[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_new
        # self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_exp

        # self._pos += len(batch["ques len"])
        batch_loss = batch_loss if num_iteration == 0 else batch_loss / num_iteration
        return num_iteration, buffer_batch_new, buffer_batch_exp, batch_loss

    def fit(self):
        print("weak?")
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]
        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        self._build_buffer_batch()
        for epo in range(self.start_epoch, epoch_nums): #
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self._scheduler_step()  #

            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s" \
                             % (self.epoch_i, loss_total / self.train_batch_nums, train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info(
                        "---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s" \
                        % (test_total, test_equ_ac, test_val_ac, test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info(
                        "---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s" \
                        % (valid_total, valid_equ_ac, valid_val_ac, valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info(
                        "---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s" \
                        % (test_total, test_equ_ac, test_val_ac, test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]''' \
                         % (self.best_valid_equ_accuracy, self.best_valid_value_accuracy, \
                            self.best_test_equ_accuracy, self.best_test_value_accuracy))

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand


class TransformerTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"], config["out_symbol2idx"][SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        #self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma=0.8)
        #self.optimizer = WarmUpScheduler(optimizer, self.config["learning_rate"], self.config["embedding_size"], self.config["warmup_steps"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _build_loss(self, symbol_size, out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight, pad)

    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def _train_batch(self, batch):
        outputs = self.model(batch["question"], batch["equation"])
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
            self.loss.eval_batch(outputs, batch_equation.view(-1))
        else:
            self.loss.eval_batch(outputs, batch["equation"].view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))
            # self.logger.info("---------- lr [%1.8f]"%(self.optimizer.get_lr()[0]))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
                            

    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0

        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

class SeqGANTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.generator_optimizer=torch.optim.Adam(self.model.generator.parameters(),\
                                                    lr=self.config["learning_rate"])
        self.discriminator_optimizer=torch.optim.Adam(self.model.discriminator.parameters(),\
                                                    lr=self.config["learning_rate"])

    def _build_loss(self, symbol_size, out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.nll_loss = NLLLoss(weight, pad)
        self.binary_loss = BinaryCrossEntropyLoss()

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.generator_optimizer.load_state_dict(check_pnt["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(check_pnt["discriminator_optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_value_accuracy = check_pnt["value_acc"]
        self.best_equ_accuracy = check_pnt["equ_acc"]

    def train_generator(self):
        print("generator pretrain...")
        for epo in range(20):
            loss_total = 0.
            self.model.generator.train()
            self.model.discriminator.eval()
            for batch_idx, batch in enumerate(self.dataloader.load_data("train")):
                self.model.zero_grad()
                outputs = self.model.generator.pre_train(batch["question"], batch["ques len"], batch["equation"])
                #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
                if self.config["share_vocab"]:
                    batch_equation = self._idx2word_2idx(batch["equation"])
                    self.nll_loss.eval_batch(outputs, batch_equation.view(-1))
                else:
                    self.nll_loss.eval_batch(outputs, batch["equation"].view(-1))
                batch_loss = self.nll_loss.get_loss()
                loss_total += batch_loss
                self.nll_loss.backward()
                self.generator_optimizer.step()
                self.nll_loss.reset()
            print("epoch [%2d] avr loss [%2.8f]" % (epo + 1, loss_total / self.train_batch_nums))

    def train_discriminator(self):
        print("discriminator pretrain...")
        for epo in range(20):
            loss_total = 0.
            self.model.generator.eval()
            self.model.discriminator.train()
            for batch_idx, batch in enumerate(self.dataloader.load_data("train")):
                self.model.zero_grad()
                output, _, _, _ = self.model.generator(batch["question"], batch["ques len"])
                pred_y = self.model.discriminator(output)
                label_y = torch.zeros_like(pred_y).to(self.config["device"])
                self.binary_loss.eval_batch(pred_y, label_y)

                if self.config["share_vocab"]:
                    batch_equation = self._idx2word_2idx(batch["equation"])
                else:
                    batch_equation = batch["equation"]
                pred_y = self.model.discriminator(batch_equation)
                label_y = torch.ones_like(pred_y).to(self.config["device"])
                self.binary_loss.eval_batch(pred_y, label_y)

                norm = self.config['l2_reg_lambda'] * (self.model.discriminator.W_O.weight.norm() + self.model.discriminator.W_O.bias.norm())
                self.binary_loss.add_norm(norm)
                loss_total += self.binary_loss.get_loss()
                self.binary_loss.backward()
                self.discriminator_optimizer.step()
                self.binary_loss.reset()
            print("epoch [%2d] avr loss [%2.8f]" % (epo + 1, loss_total / self.train_batch_nums))

    def get_reward(self, outputs, monte_carlo_outputs, token_logits):
        rewards = 0
        batch_size = outputs.size(0)
        steps = len(monte_carlo_outputs)
        for idx in range(steps):
            output = self.model.discriminator(monte_carlo_outputs[idx])
            reward = output.reshape(batch_size, -1).mean(dim=1)
            mask = outputs[:, idx] != self.config["out_pad_token"]
            reward = reward * token_logits[idx] * mask.float()
            mask_sum = mask.sum()
            if (mask_sum):
                rewards += reward.sum() / mask_sum
        return -rewards

    def _train_batch(self, batch):
        self.model.generator.train()
        self.model.discriminator.eval()
        outputs, _, monte_carlo_outputs, P = self.model.generator(batch["question"], batch["ques len"], batch["equation"])
        g_loss = self.get_reward(outputs, monte_carlo_outputs, P)

        self.model.generator.eval()
        self.model.discriminator.train()
        pred_y = self.model.discriminator(outputs)
        label_y = torch.zeros_like(pred_y).to(self.config["device"])
        self.binary_loss.eval_batch(pred_y, label_y)

        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
        else:
            batch_equation = batch["equation"]
        pred_y = self.model.discriminator(batch_equation)
        label_y = torch.ones_like(pred_y).to(self.config["device"])
        self.binary_loss.eval_batch(pred_y, label_y)

        norm = self.config['l2_reg_lambda'] * (self.model.discriminator.W_O.weight.norm() + self.model.discriminator.W_O.bias.norm())
        self.binary_loss.add_norm(norm)
        d_loss = self.binary_loss.get_loss()
        self.model.generator.train()
        self.model.discriminator.train()
        return g_loss, d_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"], batch["ques len"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        g_loss_total = 0.
        d_loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            g_batch_loss, d_batch_loss = self._train_batch(batch)
            g_loss_total += g_batch_loss
            g_batch_loss.backward()
            self.generator_optimizer.step()
            d_loss_total += d_batch_loss
            self.binary_loss.backward()
            self.discriminator_optimizer.step()
            self.binary_loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return g_loss_total, d_loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        # generator pretrain
        self.train_generator()
        # discriminator pretrain
        self.train_discriminator()
        # seqgan train
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            g_loss_total, d_loss_total, train_time_cost = self._train_epoch()
            print("epoch [%2d] avr g_loss [%2.8f] avr d_loss [%2.8f]"%(self.epoch_i,g_loss_total/self.train_batch_nums,\
                                                                        d_loss_total/self.train_batch_nums))
            print("---------- train time {}".format(train_time_cost))
            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                equation_ac, value_ac, eval_total, test_time_cost = self.evaluate()
                print("---------- test equ acc [%2.3f] | test value acc [%2.3f]" % (equation_ac, value_ac))
                print("---------- test time {}".format(test_time_cost))
                if value_ac >= self.best_value_accuracy:
                    self.best_value_accuracy = value_ac
                    self.best_equ_accuracy = equation_ac
                    self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()

    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost


class GPT2Trainer(TransformerTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_loss(config["vocab_size"], config["out_symbol2idx"][SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
    
    def _train_batch(self, batch):
        outputs, target = self.model(batch["ques_source"], batch["equ_source"])
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)

        self.loss.eval_batch(outputs, target.view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        test_out, _ = self.model(batch["ques_source"])
        target = self.model.encode_(batch["equ_source"])
        batch_size = len(target)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

class BERTGenTrainer(TransformerTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_loss(len(self.dataloader.dataset.out_symbol2idx), self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    # def _build_loss(self, symbol_size, out_pad_token):
    #     weight = torch.ones(symbol_size).to(self.config["device"])
    #     pad = out_pad_token
    #     self.loss = NLLLoss(weight, pad)

    def _train_batch(self, batch):
        # print (batch)
        outputs, target = self.model(batch["ques_source"], batch["equ_source"])
        # print ("outputs, target:", outputs.size(), target.size())
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)

        # print (outputs.size(), target.size())
        # print (outputs)
        # print (target)
        self.loss.eval_batch(outputs, target.contiguous().view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s" \
                             % (self.epoch_i, loss_total / self.train_batch_nums, train_time_cost))
                             # + "\n---------- lr [%1.8f]" % (self.optimizer.get_lr()[0]))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                self.logger.info(
                    "---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s" \
                    % (valid_total, valid_equ_ac, valid_val_ac, valid_time_cost))
                test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                self.logger.info(
                    "---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s" \
                    % (test_total, test_equ_ac, test_val_ac, test_time_cost))

                if valid_val_ac >= self.best_valid_value_accuracy:
                    self.best_valid_value_accuracy = valid_val_ac
                    self.best_valid_equ_accuracy = valid_equ_ac
                    self.best_test_value_accuracy = test_val_ac
                    self.best_test_equ_accuracy = test_equ_ac
                    self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]''' \
                         % (self.best_valid_equ_accuracy, self.best_valid_value_accuracy, \
                            self.best_test_equ_accuracy, self.best_test_value_accuracy))

    def _eval_batch(self, batch):
        test_out, _ = self.model(batch["ques_source"])
        target = batch["equ_source"]
        batch_size = len(target)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            # val_ac, equ_ac, _, _ = self.evaluator.eval_source(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            if self.config["task_type"] == TaskType.SingleEquation:

                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx],
                                                             batch["num list"][idx], batch["num stack"][idx], True)
            elif self.config["task_type"] == TaskType.MultiEquation:
                # print("test_out[idx]:", test_out[idx])
                # print("target[idx]:", target[idx])
                # print()
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx],
                                                             batch["num list"][idx], batch["num stack"][idx], True)
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

class TRNNTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["temp_symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN],config["operator_nums"])
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _build_loss(self, symbol_size, out_pad_token, operator_num):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.seq2seq_loss = NLLLoss(weight, pad)
        weight2=torch.ones(operator_num).to(self.config["device"])
        self.ans_module_loss=NLLLoss(weight2)

    def _temp_idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.temp_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_
    def _equ_idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def _idx2symbol(self,equation):
        symbols=[]
        eos_idx=self.dataloader.dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        pad_idx=self.dataloader.dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
        sos_idx=self.dataloader.dataset.temp_symbol2idx[SpecialTokens.SOS_TOKEN]
        for idx in equation:
            if idx in [eos_idx,pad_idx,sos_idx]:
                break
            symbols.append(self.dataloader.dataset.temp_idx2symbol[idx])
        return symbols
    
    def _train_batch_seq2seq(self, batch):
        outputs = self.model.seq2seq_forward(batch["question"], batch["ques len"], batch["template"])
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        if self.config["share_vocab"]:
            batch_equation = self._temp_idx2word_2idx(batch["template"])
            self.seq2seq_loss.eval_batch(outputs, batch_equation.view(-1))
        else:
            self.seq2seq_loss.eval_batch(outputs, batch["template"].view(-1))
        batch_loss = self.seq2seq_loss.get_loss()
        return batch_loss
    def _train_batch_answer_module(self, batch):
        '''seq,seq_length,num_pos, template'''
        for idx,equ in enumerate(batch["equ_source"]):
            batch["equ_source"][idx]=equ.split(" ")
        outputs,target = self.model.answer_module_forward(batch["question"], batch["ques len"], batch["num pos"], batch["equ_source"])
        for b_i in range(len(target)):
            output=torch.nn.functional.log_softmax(outputs[b_i],dim=1)
            self.ans_module_loss.eval_batch(output, target[b_i].view(-1))
        batch_loss = self.ans_module_loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        equ_acc = []
        val_acc = []
        template,equation = self.model(batch["question"], batch["ques len"],batch["num pos"])
        if self.config["share_vocab"]:
            #temp_target = self._temp_idx2word_2idx(batch["template"])
            equ_target = self._equ_idx2word_2idx(batch["equation"])
        else:
            #temp_target = batch["template"]
            equ_target = batch["equation"]
        batch_size = equ_target.size(0)
        for idx in range(batch_size):
            # test=self._idx2symbol(template[idx])
            # tar=self._idx2symbol(temp_target[idx])
            # if test==tar:
            #     equ_ac=True
            # else:
            #     equ_ac=False
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(equation[idx], equ_target[idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(equation[idx], equ_target[idx], batch["num list"][idx], batch["num stack"][idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc
    def _eval_batch_seq2seq(self,batch):
        test_out = self.model.seq2seq_forward(batch["question"], batch["ques len"])
        if self.config["share_vocab"]:
            target = self._temp_idx2word_2idx(batch["template"])
        else:
            target = batch["template"]
        batch_size = target.size(0)
        equ_acc = []
        for idx in range(batch_size):
            test=self._idx2symbol(test_out[idx])
            tar=self._idx2symbol(target[idx])
            if test==tar:
                equ_ac=True
            else:
                equ_ac=False
            equ_acc.append(equ_ac)
        return equ_acc
    def _eval_batch_answer_module(self,batch):
        for idx,equ in enumerate(batch["equ_source"]):
            batch["equ_source"][idx]=equ.split(" ")
        test_out = self.model.test_ans_module(batch["question"], batch["ques len"], batch["num pos"], batch["equ_source"])
        if self.config["share_vocab"]:
            target = self._equ_idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        for idx in range(batch_size):
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
        return val_acc
    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total_seq2seq = 0.
        loss_total_ans_module=0.
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            #train seq2seq module
            self.model.train()
            self.model.embedder.eval()
            self.model.attn_encoder.eval()
            self.model.recursivenn.eval()
            self.model.zero_grad()
            batch_loss = self._train_batch_seq2seq(batch)
            loss_total_seq2seq += batch_loss
            self.seq2seq_loss.backward()
            self.optimizer.step()
            self.seq2seq_loss.reset()
            #train answer module
            self.model.train()
            self.model.seq2seq.eval()
            self.model.zero_grad()
            batch_loss = self._train_batch_answer_module(batch)
            loss_total_ans_module += batch_loss
            self.ans_module_loss.backward()
            self.optimizer.step()
            self.ans_module_loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total_seq2seq,loss_total_ans_module, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total_seq2seq,loss_total_ans_module, train_time_cost = self._train_epoch()

            self.logger.info("epoch [%3d] avr seq2seq module loss [%2.8f] | avr answer module loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total_seq2seq/self.train_batch_nums,loss_total_ans_module/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            #batch_equ_ac = self._eval_batch_seq2seq(batch)
            #batch_val_ac = self._eval_batch_answer_module(batch)
            #batch_equ_ac = []
            batch_val_ac,batch_equ_ac=self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

class Graph2TreeTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        # self.logger.info("get group nums...")
        # self.dataloader.dataset.build_group_nums_for_graph()
    
    def _train_batch(self, batch):
        '''
        seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                UNK_TOKEN,num_start,target=None, target_length=None,max_length=30,beam_size=5
        '''
        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]

        outputs=self.model(batch["question"],batch["ques len"],batch["group nums"],batch["num list"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start,batch["equation"],batch["equ len"],UNK_TOKEN=unk)
        self.loss.eval_batch(outputs, batch["equation"], batch["equ mask"])
        batch_loss = self.loss.get_loss()
        return batch_loss
    
    def _eval_batch(self, batch):
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]
        test_out=self.model(batch["question"],batch["ques len"],batch["group nums"],batch["num list"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start)
        
        if self.config["task_type"]==TaskType.SingleEquation:
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out, batch["equation"].tolist()[0], batch["num list"][0], batch["num stack"][0])
        elif self.config["task_type"]==TaskType.MultiEquation:
            val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out, batch["equation"].tolist()[0], batch["num list"][0], batch["num stack"][0])
        else:
            raise NotImplementedError
        return [val_ac], [equ_ac]

class Graph2TreeIBMTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"],config["out_symbol2idx"][SpecialTokens.PAD_TOKEN])
        # self.logger.info("build deprel tree...")
        # self.dataloader.dataset.build_deprel_tree()

    
    def _build_loss(self, symbol_size,out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight,pad)
    
    def _build_optimizer(self):
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(),  lr=self.config["learning_rate"], weight_decay=1e-5)
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),  lr=self.config["learning_rate"])
        self.attention_optimizer = torch.optim.Adam(self.model.attention.parameters(),  lr=self.config["learning_rate"])
    
    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "attention_optimizer": self.attention_optimizer.state_dict(), 
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.decoder_optimizer.load_state_dict(check_pnt["decoder_optimizer"])
        self.attention_optimizer.load_state_dict(check_pnt["attention_optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]


    def _optimizer_step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.attention_optimizer.step()

    def _model_zero_grad(self):
        self.model.encoder.zero_grad()
        self.model.decoder.zero_grad()
        self.model.attention.zero_grad()

    def _model_train(self):
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.attention.train()

    def _model_eval(self):
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.attention.eval()

    def _train_batch(self, batch):
        outputs,target=self.model(batch["question"],batch["ques len"],batch["group nums"],batch["equation"])
        self.loss.eval_batch(outputs, target.view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss

    def _eval_batch(self, batch):
        '''seq, seq_length, group_nums, target'''
        test_out=self.model(batch["question"],batch["ques len"],batch["group nums"])
        val_acc = []
        equ_acc = []
        batch_size = len(batch["equation"])
        for idx in range(batch_size):
            if self.config["task_type"]==TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], batch["equation"][idx], batch["num list"][idx], batch["num stack"][idx])
            elif self.config["task_type"]==TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], batch["equation"][idx], batch["num list"][idx], batch["num stack"][idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc
    
    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self._model_train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self._model_zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self._optimizer_step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost
    
    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]
        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()

            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
        

    def evaluate(self, eval_set):
        self._model_eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

class MathDQNTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss()

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
    def _build_loss(self):
        self.loss = MSELoss()
    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _train_batch(self, batch):
        '''seq,seq_length,num_pos,target=None'''
        outputs = self.model(batch['question'],batch["ques len"],batch['num pos'],batch['num list'],batch['ans'],batch['equation'])
        #outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        #target = batch['equation']
        output = outputs[0]
        target = outputs[1]
        self.loss.eval_batch(output, target.view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss
    def _eval_batch(self, batch):
        '''seq,seq_length,num_pos,target=None'''
        ans_acc = self.model.predict(batch['question'],batch["ques len"],batch['num pos'],batch['num list'],batch['ans'],batch['equation'])
        val_acc=[True]*ans_acc+[False]*(len(batch['equation'])-ans_acc)
        equ_acc=[]

        return val_acc, equ_acc
    
    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.model.train()
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            #self.logger.info(self.model.training)
            loss_total, train_time_cost = self._train_epoch()
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))
            #self.logger.info(self.model.training)

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        #self.logger.info(self.model.training)
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
            
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

class HMSTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"],self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])
        self.out_pad_idx=self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["step_size"], gamma=self.config["scheduler_gamma"])
    
    def _build_loss(self,symbol_size,out_pad_token):
        weight = torch.ones(symbol_size).to(self.config["device"])
        pad = out_pad_token
        self.loss = NLLLoss(weight, pad,size_average=False)
    
    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t":self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        self.scheduler.load_state_dict(check_pnt["scheduler"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]
    
    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_


    def _train_batch(self, batch):
        if self.config["share_vocab"]:
            batch_equation = self._idx2word_2idx(batch["equation"])
        else:
            batch_equation = batch["equation"]
        outputs,_,_ = self.model(batch["spans"], batch["spans len"],batch["span num pos"],batch["word num poses"],batch["span nums"],batch["deprel tree"], batch_equation)
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        #outputs = torch.cat(outputs)
        batch_size = batch_equation.size(0)
        for step,output in enumerate(outputs):
            self.loss.eval_batch(output.contiguous().view(batch_size, -1), batch_equation[:, step].view(-1))
        batch_loss = self.loss.get_loss()
        total_target_length = (batch_equation != self.out_pad_idx).sum().item()
        batch_loss = batch_loss / total_target_length
        return batch_loss

    def _eval_batch(self, batch):
        test_out = self.model(batch["question"], batch["spans len"],batch["span num pos"],batch["word num poses"],batch["span nums"],batch["deprel tree"])
        if self.config["share_vocab"]:
            target = self._idx2word_2idx(batch["equation"])
        else:
            target = batch["equation"]
        batch_size = target.size(0)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            #val_ac, equ_ac, _, _ = self.evaluator.result(target[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx], batch["num list"][idx], batch["num stack"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if valid_val_ac >= self.best_valid_value_accuracy:
                        self.best_valid_value_accuracy = valid_val_ac
                        self.best_valid_equ_accuracy = valid_equ_ac
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
            if epo % 5 == 0:
                self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))
    
    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
            
        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def test(self):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(DatasetType.Test):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))
