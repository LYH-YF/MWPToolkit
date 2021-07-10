from logging import getLogger

import torch
import time

from mwptoolkit.trainer.abstract_trainer import AbstractTrainer
from mwptoolkit.trainer.supervised_trainer import GTSTrainer, SupervisedTrainer
from mwptoolkit.utils.enum_type import TaskType, DatasetType, SpecialTokens
from mwptoolkit.utils.utils import time_since


class GTSWeakTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self.supervising_mode = config["supervising_mode"]
        self._build_optimizer()
 

    def _build_buffer_batch(self):
        self._buffer_batches = [[] for i in range(self.dataloader.trainset_nums)]
        self._buffer_batches_exp = [[] for i in range(self.dataloader.trainset_nums)]
   
    def _train_epoch(self):
        epoch_start_time = time.time() #
        loss_total = 0.  #
      
        self.mask_flag = False #
        self._pos = 0 #
        self.epo_iteration = 0
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.train()
            self.model.zero_grad()
            if self.epoch_i == 1 and self.batch_idx <= 2: #
                self.mask_flag = True
            buffer_batches_train = self._buffer_batches[self._pos: self._pos + len(batch["ques len"])]
            buffer_batches_train_exp = self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])]
            iterations, buffer_batch_new, buffer_batch_exp, batch_loss = self._train_batch(batch, buffer_batches_train, buffer_batches_train_exp)
            loss_total += batch_loss
            self.epo_iteration += iterations
            self._buffer_batches[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_new
            self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_exp
            self._pos += len(batch["ques len"])
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def _train_batch(self, batch, buffer_batches_train, buffer_batches_train_exp):
        optimizer = {
            "embedder_optimizer": self.embedder_optimizer,
            "encoder_optimizer": self.encoder_optimizer,
            "decoder_optimizer": self.decoder_optimizer,
            "node_generater_optimizer": self.node_generater_optimizer,
            "merge_optimizer": self.merge_optimizer
        }
        num_iteration, buffer_batch_new, buffer_batch_exp, batch_loss = self.model.weakly_train(batch, buffer_batches_train, buffer_batches_train_exp, self.dataloader, self.epoch_i - 1, 
        self.mask_flag, self.supervising_mode, optimizer)
        return num_iteration, buffer_batch_new, buffer_batch_exp, batch_loss
    
    def fit(self):
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


class WeaklySupervisedTrainer(SupervisedTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self.supervising_mode = config["supervising_mode"]
        self._build_optimizer()
 

    def _build_buffer_batch(self):
        self._buffer_batches = [[] for i in range(self.dataloader.trainset_nums)]
        self._buffer_batches_exp = [[] for i in range(self.dataloader.trainset_nums)]
   
    def _train_epoch(self):
        epoch_start_time = time.time() #
        loss_total = 0.  #
      
        self.mask_flag = False #
        self._pos = 0 #
        self.epo_iteration = 0
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.train()
            self.model.zero_grad()
            if self.epoch_i == 1 and self.batch_idx <= 2: #
                self.mask_flag = True
            buffer_batches_train = self._buffer_batches[self._pos: self._pos + len(batch["ques len"])]
            buffer_batches_train_exp = self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])]
            iterations, buffer_batch_new, buffer_batch_exp, batch_loss = self._train_batch(batch, buffer_batches_train, buffer_batches_train_exp)
            loss_total += batch_loss
            self.epo_iteration += iterations
            self._buffer_batches[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_new
            self._buffer_batches_exp[self._pos: self._pos + len(batch["ques len"])] = buffer_batch_exp
            self._pos += len(batch["ques len"])
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def _train_batch(self, batch, buffer_batches_train, buffer_batches_train_exp):
        optimizer = self.optimizer
        num_iteration, buffer_batch_new, buffer_batch_exp, batch_loss = self.model.weakly_train(batch, buffer_batches_train, buffer_batches_train_exp, self.dataloader, self.epoch_i - 1, 
        self.mask_flag, self.supervising_mode, optimizer)
        return num_iteration, buffer_batch_new, buffer_batch_exp, batch_loss
    
    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]
        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        self._build_buffer_batch()
        for epo in range(self.start_epoch, epoch_nums): #
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()

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
