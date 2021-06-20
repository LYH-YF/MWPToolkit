import time

import torch
import numpy as np

from mwptoolkit.trainer.abstract_trainer import AbstractTrainer
from mwptoolkit.trainer.template_trainer import TemplateTrainer
from mwptoolkit.utils.enum_type import TaskType, DatasetType, SpecialTokens
from mwptoolkit.utils.utils import time_since


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        #self._build_loss(config["symbol_size"], self.dataloader.dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN])

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
            "fold_t": self.config["fold_t"]
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
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss
    
    def _eval_batch(self, batch):
        test_out, target = self.model.model_test(batch)
        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx])
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
            self.optimizer.step()
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
            "fold_t": self.config["fold_t"]
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

    def _train_batch(self, batch):
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss

    def _eval_batch(self, batch):
        test_out, target = self.model.model_test(batch)

        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx])
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
            self._optimizer_step()
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
            pass

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


class MultiEncDecTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

    def _build_optimizer(self):
        # optimizer
        # self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.numencoder_optimizer = torch.optim.Adam(self.model.numencoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.predict_optimizer = torch.optim.Adam(self.model.predict.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.decoder_optimizer = torch.optim.Adam(
            [
                {'params': self.model.decoder.parameters()}, \
                {'params': self.model.out.parameters()}, \
                {'params': self.model.out_embedder.parameters()}\
            ],
            self.config["learning_rate"], \
            weight_decay=self.config["weight_decay"]
        )
        self.generate_optimizer = torch.optim.Adam(self.model.generate.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.merge_optimizer = torch.optim.Adam(self.model.merge.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        #self.optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # scheduler
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.numencoder_scheduler = torch.optim.lr_scheduler.StepLR(self.numencoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.predict_scheduler = torch.optim.lr_scheduler.StepLR(self.predict_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.generate_scheduler = torch.optim.lr_scheduler.StepLR(self.generate_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(self.merge_optimizer, step_size=self.config["step_size"], gamma=0.5)

    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        #self.optimizer.load_state_dict(check_pnt['optimizer'])
        self.numencoder_optimizer.load_state_dict(check_pnt["numencoder_optimizer"])
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.predict_optimizer.load_state_dict(check_pnt['predict_optimizer'])
        self.decoder_optimizer.load_state_dict(check_pnt["decoder_optimizer"])
        self.generate_optimizer.load_state_dict(check_pnt["generate_optimizer"])
        self.merge_optimizer.load_state_dict(check_pnt["merge_optimizer"])
        #load parameter of scheduler
        #self.scheduler.load_state_dict(check_pnt['scheduler'])
        self.encoder_scheduler.load_state_dict(check_pnt["encoder_scheduler"])
        self.numencoder_scheduler.load_state_dict(check_pnt["numencoder_scheduler"])
        self.predict_scheduler.load_state_dict(check_pnt['predict_scheduler'])
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

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "numencoder_optimizer": self.numencoder_optimizer.state_dict(),
            "predict_optimizer": self.predict_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_optimizer": self.generate_optimizer.state_dict(),
            "merge_optimizer": self.merge_optimizer.state_dict(),
            "encoder_scheduler": self.encoder_scheduler.state_dict(),
            "numencoder_scheduler": self.numencoder_scheduler.state_dict(),
            "predict_scheduler": self.predict_scheduler.state_dict(),
            "decoder_scheduler": self.decoder_scheduler.state_dict(),
            "generate_scheduler": self.generate_scheduler.state_dict(),
            "merge_scheduler": self.merge_scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t": self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _scheduler_step(self):
        #self.scheduler.step()
        self.encoder_scheduler.step()
        self.numencoder_scheduler.step()
        self.predict_scheduler.step()
        self.decoder_scheduler.step()
        self.generate_scheduler.step()
        self.merge_scheduler.step()

    def _optimizer_step(self):
        #self.optimizer.step()
        self.encoder_optimizer.step()
        self.numencoder_optimizer.step()
        self.predict_optimizer.step()
        self.decoder_optimizer.step()
        self.generate_optimizer.step()
        self.merge_optimizer.step()

    def _train_batch(self, batch):
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss

    def _eval_batch(self, batch):
        out_type, test_out, target = self.model.model_test(batch)

        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation and out_type == 'tree':
                val_ac, equ_ac, _, _ = self.evaluator.prefix_result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.SingleEquation and out_type == 'attn':
                val_ac, equ_ac, _, _ = self.evaluator.postfix_result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation and out_type == 'tree':
                val_ac, equ_ac, _, _ = self.evaluator.prefix_result_multi(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation and out_type == 'attn':
                val_ac, equ_ac, _, _ = self.evaluator.postfix_result_multi(test_out[idx], target[idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc


class Graph2TreeTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)


class TreeLSTMTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()

    def _build_optimizer(self):
        # optimizer
        self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.decoder_optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.node_generater_optimizer = torch.optim.Adam(self.model.node_generater.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # scheduler
        self.embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.node_generater_scheduler = torch.optim.lr_scheduler.StepLR(self.node_generater_optimizer, step_size=self.config["step_size"], gamma=0.5)

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "embedder_optimizer": self.embedder_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_optimizer": self.node_generater_optimizer.state_dict(),
            "embedder_scheduler": self.embedder_scheduler.state_dict(),
            "encoder_scheduler": self.encoder_scheduler.state_dict(),
            "decoder_scheduler": self.decoder_scheduler.state_dict(),
            "generate_scheduler": self.node_generater_scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t": self.config["fold_t"]
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
        # load parameter of scheduler
        self.embedder_scheduler.load_state_dict(check_pnt["embedder_scheduler"])
        self.encoder_scheduler.load_state_dict(check_pnt["encoder_scheduler"])
        self.decoder_scheduler.load_state_dict(check_pnt["decoder_scheduler"])
        self.node_generater_scheduler.load_state_dict(check_pnt["generate_scheduler"])
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

    def _optimizer_step(self):
        self.embedder_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.node_generater_optimizer.step()

    def _train_batch(self, batch):
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss

    def _eval_batch(self, batch):
        test_out, target = self.model.model_test(batch)

        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx])
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
            self._optimizer_step()
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


class SAUSolverTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

    def _train_batch(self, batch):
        try:
            batch_loss = self.model.calculate_loss(batch)
        except:
            print(batch['id'])
        return batch_loss

    def _eval_batch(self, batch):
        try:
            test_out, target = self.model.model_test(batch)
        except:
            print(batch['id'])

        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx])
            else:
                raise NotImplementedError
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc, equ_acc


class TRNNTrainer(SupervisedTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.config["learning_rate"])
        # self.seq2seq_optimizer = torch.optim.Adam(
        #     [
        #         {'params': self.model.seq2seq_in_embedder.parameters()}, \
        #         {'params': self.model.seq2seq_out_embedder.parameters()}, \
        #         {'params': self.model.seq2seq_encoder.parameters()}, \
        #         {'params': self.model.seq2seq_decoder.parameters()}, \
        #         {'params': self.model.seq2seq_gen_linear.parameters()}\
        #     ],
        #     self.config["learning_rate"]
        # )

        # self.answer_module_optimizer = torch.optim.Adam(
        #     [
        #         {'params': self.model.answer_in_embedder.parameters()}, \
        #         {'params': self.model.answer_encoder.parameters()}, \
        #         {'params': self.model.answer_rnn.parameters()}\
        #     ], 
        #     self.config["learning_rate"]
        # )

    # def _save_checkpoint(self):
    #     check_pnt = {
    #         "model": self.model.state_dict(),
    #         "seq2seq_optimizer": self.seq2seq_optimizer.state_dict(),
    #         "answer_module_optimizer": self.answer_module_optimizer.state_dict(),
    #         "start_epoch": self.epoch_i,
    #         "best_valid_value_accuracy": self.best_valid_value_accuracy,
    #         "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
    #         "best_test_value_accuracy": self.best_test_value_accuracy,
    #         "best_test_equ_accuracy": self.best_test_equ_accuracy,
    #         "best_folds_accuracy": self.best_folds_accuracy,
    #         "fold_t": self.config["fold_t"]
    #     }
    #     torch.save(check_pnt, self.config["checkpoint_path"])

    # def _load_checkpoint(self):
    #     #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
    #     check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
    #     # load parameter of model
    #     self.model.load_state_dict(check_pnt["model"])
    #     # load parameter of optimizer
    #     self.seq2seq_optimizer.load_state_dict(check_pnt["optimizer"])
    #     self.answer_module_optimizer.load_state_dict(check_pnt["answer_module_optimizer"])
    #     # other parameter
    #     self.start_epoch = check_pnt["start_epoch"]
    #     self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
    #     self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
    #     self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
    #     self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
    #     self.best_folds_accuracy = check_pnt["best_folds_accuracy"]
    
    def _train_seq2seq_batch(self, batch):
        batch_loss = self.model.seq2seq_calculate_loss(batch)
        return batch_loss
    def _train_ans_batch(self, batch):
        batch_loss = self.model.ans_module_calculate_loss(batch)
        return batch_loss

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total_seq2seq = 0.
        loss_total_ans_module = 0.
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            # first stage
            self.model.seq2seq_in_embedder.train()
            self.model.seq2seq_out_embedder.train()
            self.model.seq2seq_encoder.train()
            self.model.seq2seq_decoder.train()
            self.model.seq2seq_gen_linear.train()
            self.model.answer_in_embedder.eval()
            self.model.answer_encoder.eval()
            self.model.answer_rnn.eval()
            self.model.zero_grad()
            batch_seq2seq_loss = self._train_seq2seq_batch(batch)
            self.optimizer.step()
            # second stage
            self.model.seq2seq_in_embedder.eval()
            self.model.seq2seq_out_embedder.eval()
            self.model.seq2seq_encoder.eval()
            self.model.seq2seq_decoder.eval()
            self.model.seq2seq_gen_linear.eval()
            self.model.answer_in_embedder.train()
            self.model.answer_encoder.train()
            self.model.answer_rnn.train()
            self.model.zero_grad()
            batch_ans_module_loss = self._train_ans_batch(batch)
            loss_total_seq2seq += batch_seq2seq_loss
            loss_total_ans_module += batch_ans_module_loss
            #self.seq2seq_optimizer.step()
            #self.answer_module_optimizer.step()
            self.optimizer.step()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total_seq2seq, loss_total_ans_module, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1
        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total_seq2seq, loss_total_ans_module, train_time_cost = self._train_epoch()

            self.logger.info("epoch [%3d] avr seq2seq module loss [%2.8f] | avr answer module loss [%2.8f] | train time %s"\
                                %(self.epoch_i,loss_total_seq2seq/self.train_batch_nums,loss_total_ans_module/self.train_batch_nums,train_time_cost))
            self.logger.info("target wrong: {} target total: {}".format(self.model.wrong, self.dataloader.trainset_nums))
            self.model.wrong=0
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


class SalignedTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()

        self.min_NUM = self.dataloader.dataset.out_symbol2idx['NUM_0']
        #print(self.dataloader.dataset.out_symbol2idx); exit()
        self.do_addeql = False if '<BRG>' in self.dataloader.dataset.out_symbol2idx else True
        max_NUM = list(self.dataloader.dataset.out_symbol2idx.keys())[-2]
        self.max_NUM = self.dataloader.dataset.out_symbol2idx[max_NUM]
        self.ADD = self.dataloader.dataset.out_symbol2idx['+']
        self.POWER = self.dataloader.dataset.out_symbol2idx['^']
        self.min_CON = self.N_OPS = self.POWER + 1
        self.UNK = self.dataloader.dataset.out_symbol2idx['<UNK>']
        self.max_CON = self.min_NUM - 1
        self.constant = list(self.dataloader.dataset.out_symbol2idx.keys())[self.min_CON: self.min_NUM]
        #print('self.constant', self.min_NUM, self.max_NUM, self.ADD, self.POWER, self.min_CON, self.max_CON, self.constant); exit()

        model.encoder.initialize_fix_constant(len(self.constant), self.model._device)

    def _build_optimizer(self):
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"],
                                                   weight_decay=self.config["weight_decay"])
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["step_size"],
                                                               gamma=0.5)

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        print('load checkpoint ...', self.config["checkpoint_path"]); #exit()
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        # print('check_pnt["model"]', check_pnt["model"].keys()) #check_pnt["model"] #
        pre_trained_dict = check_pnt["model"] #{k: check_pnt["model"][k] for k in check_pnt["model"] if ('decoder.op_selector' not in k and k != '_op_loss.weight')}
        self.model.load_state_dict(pre_trained_dict, strict=False)

    def _scheduler_step(self):
        self.scheduler.step()

    def _optimizer_step(self):
        self.optimizer.step()

    def _model_zero_grad(self):
        self.model.zero_grad()

    def _model_train(self):
        self.model.train()

    def _model_eval(self):
        self.model.eval()

    def _train_batch(self, batch, N_OPS):
        '''
        seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                UNK_TOKEN,num_start,target=None, target_length=None,max_length=30,beam_size=5
        '''
        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums = [self.dataloader.dataset.out_symbol2idx[symbol] for symbol in
                         self.dataloader.dataset.generate_list]

        order = torch.sort(batch['ques len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]

        #print('Saligned batch', len(batch), batch.keys())
        batch_size = len(batch['ques len'])
        # print(batch['ans'][0], batch['id'], batch['constants'][0], batch['constant_indices'][0])
        # print(batch['text_len'][0], batch['text'].size(), batch['operations'][0], batch['op_len'][0], batch['indice'][0])
        #print(batch['ans'][0], batch["id"][0], batch['num list'][0], batch["num pos"][0]);
        #print(batch["ques len"][0], batch["equation"][0], batch['equ len'][0])
        #exit()
        #print('forward')
        outputs, loss = self.model(batch["question"], batch["equation"], batch["ques len"],
                                   batch["num pos"], batch['num list'], self.constant, batch["equ len"], N_OPS)
        #print('train_out', loss, outputs[0], batch["raw_equation"].tolist()[0]); #exit()
        #print(batch["equ len"], self.evaluator.out_expression_list(batch["raw_equation"].tolist()[0],
        #                                       batch["num list"][0], copy.deepcopy(batch["num stack"][0])))
        batch_acc = []
        for i in range(batch_size):
            #print('equation', list(batch["equation"][i][1:].cpu().numpy()),  outputs[i]); exit()
            batch_acc += [int(list(batch["equation"][i][1:].cpu().numpy())[:len(outputs[i])] == outputs[i])]
        # outputs, loss = self.model(batch["question"], batch["ques len"], batch["num stack"], batch["num size"], \
        #                      generate_nums, batch["num pos"], num_start, batch["equation"], batch["equ len"],
        #                      UNK_TOKEN=unk)
        batch_loss = loss #.get_loss()
        #print('batch_acc', batch_acc)
        return batch_loss, batch_acc

    def _eval_batch(self, batch, N_OPS):
        order = torch.sort(batch['ques len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]
        # if batch["id"]==[80313]:
        #     print(1)
        # else:
        #     return [False], [False]
        #print(batch["question"])
        test_out = self.model.predict(batch["question"], batch["equation"], batch["ques len"],
                              batch["num pos"], batch['num list'], self.constant, batch["equ len"], N_OPS)
        #print('predict_test_out', test_out[0]["equations_index"]);
        #print(test_out)
        #print('test_out', test_out[0]["equations"], batch["equation"].tolist()[0]); #exit()
        val_acs, equ_acs = [], []
        for i in range(len(test_out)):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[i]["equations"], batch["raw_equation"].tolist()[i], batch["num list"][i],
                                                             batch["num stack"][i])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[i]["equations"], batch["raw_equation"].tolist()[i],
                                                                   batch["num list"][i], batch["num stack"][i]) #, i==0
            else:
                raise NotImplementedError
            val_acs += [val_ac]
            equ_acs += [equ_ac]
        return val_acs, equ_acs

    def adjust_equ(self, op_target, eq_len, num_list):
        batch_size, batch_len = op_target.size()
        # change NUM
        # target_mask = torch.ge(op_target, self.min_NUM) * torch.le(op_target, self.max_NUM).to(torch.long)
        # op_target = (op_target + self.UNK - self.min_NUM + 4) * target_mask + op_target * (1 - target_mask)
        # change constants
        target_mask = torch.ge(op_target, self.min_CON) * torch.le(op_target, self.max_NUM).to(torch.long)
        op_target = (op_target + 3) * target_mask + op_target * (1 - target_mask)
        # change unk
        target_mask = torch.eq(op_target, self.UNK).to(torch.long)
        op_target = (self.min_NUM + 3) * target_mask + op_target * (1 - target_mask)
        # change +/-/*//
        target_mask = torch.ge(op_target, self.ADD) * torch.le(op_target, self.POWER - 1).to(torch.long)
        op_target = (op_target + 2) * target_mask + op_target * (1 - target_mask)
        # change padding
        #print(eq_len, num_list)
        target_mask = torch.tensor([[1]*eq_len[b]+[0]*(batch_len-eq_len[b]) for b in range(batch_size)]).to(torch.long).to(self.model._device)
        op_target = op_target * target_mask
        # attach prefix/postfix
        batch_size, _ = op_target.size()
        #if self.do_addeql:
        eq_postfix = torch.zeros((batch_size, 1), dtype=torch.long).to(self.model._device) + 2
        op_target = torch.cat([op_target, eq_postfix], dim=1)
        op_target.scatter_(1, torch.tensor([[idx] for idx in eq_len]).to(self.model._device), self.model.EQL)
        #op_target[torch.arange(batch_size).unsqueeze(1), eq_len] = self.model.EQL
        #print('op_target', op_target[:3, :10])
        gen_var_prefix = [self.min_NUM + len(num) + 3 for num in num_list]
        #print('gen_var_prefix', self.max_NUM, num_list, gen_var_prefix)
        gen_var_prefix = torch.tensor(gen_var_prefix, dtype=torch.long).unsqueeze(1).to(self.model._device)
        #gen_var_prefix = torch.zeros((batch_size, 1), dtype=torch.long).to(self.model._device) + 14 #self.max_NUM + 4
        x_prefix = torch.zeros((batch_size, 1), dtype=torch.long).to(self.model._device) + self.model.GEN_VAR
        op_target = torch.cat([x_prefix, gen_var_prefix, op_target], dim=1)
        #if self.do_addeql:
        eq_len = [(idx + 3) for idx in eq_len]
        # else:
        #     eq_len = [(idx + 2) for idx in eq_len]

        return op_target, eq_len

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        acc_total = 0.
        N_OPS = self.N_OPS
        #print('self.dataset.out_symbol2idx', self.dataloader.dataset.out_symbol2idx)
        self._model_train()

        #print('min_NUM, max_NUM', min_NUM, max_NUM, N_OPS); #exit()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            #print('batch_idx', batch_idx)
            #if batch_idx != 0: continue
            #if batch_idx >= 50: continue
            batch["raw_equation"] = batch["equation"].clone()
            self.batch_idx = batch_idx + 1
            self._model_zero_grad()
            # if self.batch_idx==42:
            #     continue
            if True:
                #print('before adjust', batch["equation"][:3, :30], batch['equ len'])
                batch["equation"], batch['equ len'] = self.adjust_equ(batch["raw_equation"], batch['equ len'], batch['num list'])
                #print('after adjust', batch["equation"][:3, :30], batch['equ len']); #exit()
                batch_loss, batch_acc = self._train_batch(batch, N_OPS)
                acc_total += np.sum(batch_acc)
                loss_total += batch_loss
                batch_loss.backward()
                self._optimizer_step()
            else: #except IndexError:
                pass #print("except IndexError")
            #self.loss.reset()
        #exit()
        #print('acc_total', acc_total)
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost
        # print("epoch [%2d]avr loss [%2.8f]"%(self.epoch_i,loss_total /self.batch_nums))
        # print("epoch train time {}".format(time_since(time.time() -epoch_start_time)))

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

            self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s" \
                             % (self.epoch_i, loss_total / self.train_batch_nums, train_time_cost))

            if True: #epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    #print('k_fold'); exit() I changed this !!!
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Train)

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

    def evaluate(self, eval_set):
        self._model_eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        #print('evaluation set', eval_set)
        for batch_idx, batch in enumerate(self.dataloader.load_data(eval_set)):
            #print(batch_idx)
            if batch_idx >= 100: continue
            #print('batch', batch.encode('utf8'))
            batch["raw_equation"] = batch["equation"].clone()
            batch["equation"], batch['equ len'] = self.adjust_equ(batch["equation"], batch['equ len'], batch['num list'])
            batch_val_ac, batch_equ_ac = self._eval_batch(batch, self.N_OPS)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
            #print('len(batch_val_ac)', batch_val_ac)

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
            batch_val_ac, batch_equ_ac = self._eval_batch(batch, self.N_OPS)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s" \
                         % (eval_total, equation_ac / eval_total, value_ac / eval_total, test_time_cost))


