# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 22:14:27
# @File: template_trainer.py


import time

from mwptoolkit.trainer.abstract_trainer import AbstractTrainer
from mwptoolkit.utils.enum_type import DatasetType
from mwptoolkit.utils.utils import time_since


class TemplateTrainer(AbstractTrainer):
    r"""template trainer.
    
    you need implement:

    TemplateTrainer._build_optimizer()

    TemplateTrainer._save_checkpoint()

    TemplateTrainer._load_checkpoint()

    TemplateTrainer._train_batch()

    TemplateTrainer._eval_batch()
    """
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
    
    def _build_optimizer(self):
        raise NotImplementedError

    def _save_checkpoint(self):
        raise NotImplementedError

    def _load_checkpoint(self):
        raise NotImplementedError

    def _build_loss(self):
        raise NotImplementedError

    def _idx2word_2idx(self):
        raise NotImplementedError

    def _train_batch(self, batch):
        raise NotImplementedError

    def _eval_batch(self, batch):
        raise NotImplementedError

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
