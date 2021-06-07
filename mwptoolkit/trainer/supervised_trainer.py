import time

import torch

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
        #outputs = self.model(batch["question"], batch["ques len"], batch["equation"])
        batch_loss = self.model.calculate_loss(batch)
        #outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        # if self.config["share_vocab"]:
        #     batch_equation = self._idx2word_2idx(batch["equation"])
        #     self.loss.eval_batch(outputs, batch_equation.view(-1))
        # else:
        #     self.loss.eval_batch(outputs, batch["equation"].view(-1))
        # batch_loss = self.loss.get_loss()
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


class MuitiEncDecTrainer(TemplateTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

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


class Graph2TreeTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

class TreeLSTMTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
    def _build_optimizer(self):
        # optimizer
        self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"],
                                                   weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"],
                                                  weight_decay=self.config["weight_decay"])
        self.decoder_optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"],
                                                  weight_decay=self.config["weight_decay"])
        self.node_generater_optimizer = torch.optim.Adam(self.model.node_generater.parameters(),
                                                         self.config["learning_rate"],
                                                         weight_decay=self.config["weight_decay"])
        # scheduler
        self.embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.embedder_optimizer,
                                                                  step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer,
                                                                 step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer,
                                                                 step_size=self.config["step_size"], gamma=0.5)
        self.node_generater_scheduler = torch.optim.lr_scheduler.StepLR(self.node_generater_optimizer,
                                                                        step_size=self.config["step_size"], gamma=0.5)

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
            # self.model.train()
            # loss_total, train_time_cost = self._train_epoch()
            # self._scheduler_step()

            # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
            #                     %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

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
