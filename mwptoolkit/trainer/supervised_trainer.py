import time

import torch
from ray import tune

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

    def param_search(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
            #                     %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                # self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                #                 %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                tune.report(accuracy=test_val_ac)

                # self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                #                 %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                # if valid_val_ac >= self.best_valid_value_accuracy:
                #     self.best_valid_value_accuracy = valid_val_ac
                #     self.best_valid_equ_accuracy = valid_equ_ac
                #     self.best_test_value_accuracy = test_val_ac
                #     self.best_test_equ_accuracy = test_equ_ac
                #     self._save_model()
            # if epo % 5 == 0:
            #     self._save_checkpoint()
        # self.logger.info('''training finished.
        #                     best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
        #                     best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
        #                     %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
        #                         self.best_test_equ_accuracy,self.best_test_value_accuracy))


class GTSTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()

    def _build_optimizer(self):
        # optimizer
        #self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        #self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(
            [
                {'params': self.model.embedder.parameters()}, \
                {'params': self.model.encoder.parameters()}
            ],
            self.config["learning_rate"]
        )
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.node_generater_optimizer = torch.optim.Adam(self.model.node_generater.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.merge_optimizer = torch.optim.Adam(self.model.merge.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # scheduler
        #self.embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.node_generater_scheduler = torch.optim.lr_scheduler.StepLR(self.node_generater_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(self.merge_optimizer, step_size=self.config["step_size"], gamma=0.5)

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_optimizer": self.node_generater_optimizer.state_dict(),
            "merge_optimizer": self.merge_optimizer.state_dict(),
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
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.decoder_optimizer.load_state_dict(check_pnt["decoder_optimizer"])
        self.node_generater_optimizer.load_state_dict(check_pnt["generate_optimizer"])
        self.merge_optimizer.load_state_dict(check_pnt["merge_optimizer"])
        #load parameter of scheduler
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
        self.encoder_scheduler.step()
        self.decoder_scheduler.step()
        self.node_generater_scheduler.step()
        self.merge_scheduler.step()

    def _optimizer_step(self):
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

    def param_search(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self._scheduler_step()
            # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
            #                     %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                # self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                #                 %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                tune.report(accuracy=test_val_ac)


class MultiEncDecTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

    def _build_optimizer(self):
        # optimizer
        # self.embedder_optimizer = torch.optim.Adam(self.model.embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.numencoder_optimizer = torch.optim.Adam(self.model.numencoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.predict_optimizer = torch.optim.Adam(self.model.predict.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
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
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
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

    def param_search(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            self._scheduler_step()
            # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
            #                     %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                # self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                #                 %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                tune.report(accuracy=test_val_ac)


class SAUSolverTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

    def _train_batch(self, batch):
        try:
            batch_loss = self.model.calculate_loss(batch)
        except:
            self.logger.info('something wrong at data with id = {}'.format(batch['id']))
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
        #self.optimizer = torch.optim.Adam(self.model.parameters(),self.config["learning_rate"])
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.model.seq2seq_in_embedder.parameters()}, \
                {'params': self.model.seq2seq_out_embedder.parameters()}, \
                {'params': self.model.seq2seq_encoder.parameters()}, \
                {'params': self.model.seq2seq_decoder.parameters()}, \
                {'params': self.model.seq2seq_gen_linear.parameters()}\
            ],
            self.config["seq2seq_learning_rate"]
        )

        self.answer_module_optimizer = torch.optim.SGD(
            [
                {'params': self.model.answer_in_embedder.parameters()}, \
                {'params': self.model.answer_encoder.parameters()}, \
                {'params': self.model.answer_rnn.parameters()}\
            ],
            self.config["ans_learning_rate"],
            momentum=0.9
        )

    def seq2seq_train(self):
        self.model.seq2seq_in_embedder.train()
        self.model.seq2seq_out_embedder.train()
        self.model.seq2seq_encoder.train()
        self.model.seq2seq_decoder.train()
        self.model.seq2seq_gen_linear.train()
        self.model.answer_in_embedder.eval()
        self.model.answer_encoder.eval()
        self.model.answer_rnn.eval()

    def ans_train(self):
        self.model.seq2seq_in_embedder.eval()
        self.model.seq2seq_out_embedder.eval()
        self.model.seq2seq_encoder.eval()
        self.model.seq2seq_decoder.eval()
        self.model.seq2seq_gen_linear.eval()
        self.model.answer_in_embedder.train()
        self.model.answer_encoder.train()
        self.model.answer_rnn.train()

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
            self.seq2seq_train()
            self.model.zero_grad()
            batch_seq2seq_loss = self._train_seq2seq_batch(batch)
            self.optimizer.step()
            # second stage
            self.ans_train()
            self.model.zero_grad()
            batch_ans_module_loss = self._train_ans_batch(batch)
            loss_total_seq2seq += batch_seq2seq_loss
            loss_total_ans_module += batch_ans_module_loss
            #self.seq2seq_optimizer.step()
            #self.answer_module_optimizer.step()
            self.answer_module_optimizer.step()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total_seq2seq, loss_total_ans_module, epoch_time_cost

    def _eval_batch(self, batch, x=0):
        test_out, target, temp_out, temp_tar, equ_out, equ_tar = self.model.model_test(batch)
        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        temp_acc = []
        equs_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target[idx])
            else:
                raise NotImplementedError

            equ_acc.append(equ_ac)
            val_acc.append(val_ac)

            if temp_out[idx] == temp_tar[idx]:
                temp_acc.append(True)
            else:
                temp_acc.append(False)
            if equ_out[idx] == equ_tar[idx]:
                equs_acc.append(True)
            else:
                equs_acc.append(False)
            if x:
                self.logger.info('{}\n{}\n{} {} {}\n{} {} {}'.format([batch["ques source 1"][idx]],[batch["ques source"][idx]],\
                    equ_out[idx],temp_out[idx],test_out[idx],\
                    equ_tar[idx],temp_tar[idx],target[idx]))

        return val_acc, equ_acc, temp_acc, equs_acc

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
            self.model.wrong = 0
            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, template_ac, equation_ac, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | seq2seq module acc [%2.3f] | answer module acc [%2.3f]"\
                                    %(test_total,template_ac,equation_ac))
                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    # if test_val_ac >= self.best_test_value_accuracy:
                    #     self.best_test_value_accuracy = test_val_ac
                    #     self.best_test_equ_accuracy = test_equ_ac
                    #     self._save_model()
                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, _, _, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, _, _, test_total, test_time_cost = self.evaluate(DatasetType.Test)

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
        #self.test(DatasetType.Test)
        #self.test(DatasetType.Train)
        self.logger.info('''training finished.
                            best valid result: equation accuracy [%2.3f] | value accuracy [%2.3f]
                            best test result : equation accuracy [%2.3f] | value accuracy [%2.3f]'''\
                            %(self.best_valid_equ_accuracy,self.best_valid_value_accuracy,\
                                self.best_test_equ_accuracy,self.best_test_value_accuracy))

    def evaluate(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        template_ac = 0
        equations_ac = 0
        eval_total = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac, batch_temp_acc, batch_equs_acc = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            template_ac += batch_temp_acc.count(True)
            equations_ac += batch_equs_acc.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total,\
                template_ac / eval_total, equations_ac / eval_total,\
                eval_total, test_time_cost

    def test(self, type):
        self._load_model()
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        ans_acc = 0
        test_start_time = time.time()

        for batch in self.dataloader.load_data(type):
            batch_val_ac, batch_equ_ac, batch_temp_acc, batch_equs_acc = self._eval_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            ans_acc += batch_equs_acc.count(True)
            eval_total += len(batch_val_ac)
        test_time_cost = time_since(time.time() - test_start_time)
        # self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
        #                         %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))
        self.logger.info("test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                %(eval_total,equation_ac/eval_total,value_ac/eval_total,test_time_cost))

    def param_search(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]

        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            seq2seq_loss_total, _, train_time_cost = self._train_epoch()
            # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s"\
            #                     %(self.epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                valid_equ_ac, valid_val_ac, _, _, valid_total, valid_time_cost = self.evaluate(DatasetType.Valid)

                # self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                #                 %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                test_equ_ac, test_val_ac, _, acc, test_total, test_time_cost = self.evaluate(DatasetType.Test)

                tune.report(accuracy=test_val_ac)

class SalignedTrainer(SupervisedTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()

        if config["resume"]:
            self._load_checkpoint()

    def _build_optimizer(self):
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["step_size"], gamma=0.5)

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
            "fold_t": self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"], map_location=self.config["map_location"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        #load parameter of scheduler
        self.scheduler.load_state_dict(check_pnt["scheduler"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_value_accuracy = check_pnt["best_valid_value_accuracy"]
        self.best_valid_equ_accuracy = check_pnt["best_valid_equ_accuracy"]
        self.best_test_value_accuracy = check_pnt["best_test_value_accuracy"]
        self.best_test_equ_accuracy = check_pnt["best_test_equ_accuracy"]
        self.best_folds_accuracy = check_pnt["best_folds_accuracy"]

    def _scheduler_step(self):
        self.scheduler.step()

    def _optimizer_step(self):
        self.optimizer.step()

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
        target_mask = torch.tensor([[1] * eq_len[b] + [0] * (batch_len - eq_len[b]) for b in range(batch_size)]).to(torch.long).to(self.model._device)
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

    def _train_batch(self, batch):
        order = torch.sort(batch['ques len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss

    def _eval_batch(self, batch):
        order = torch.sort(batch['ques len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]

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
        #print(self.dataloader.dataset.out_symbol2idx); #exit()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            #if batch_idx >= 100: continue
            #print('batch_idx', batch_idx)
            batch["raw_equation"] = batch["equation"].clone()
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            batch_loss.backward()
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
        for batch_idx, batch in enumerate(self.dataloader.load_data(eval_set)):
            if batch_idx >= 3000: continue
            batch["raw_equation"] = batch["equation"].clone()
            # batch["equation"], batch['equ len'] = self.adjust_equ(batch["raw_equation"], batch['equ len'],
            #                                                       batch['num list'])
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


class HMSTrainer(GTSTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["step_size"], gamma=self.config["scheduler_gamma"])

    def _optimizer_step(self):
        self.optimizer.step()

    def _scheduler_step(self):
        self.scheduler.step()

    def _load_checkpoint(self):
        #check_pnt = torch.load(self.config["checkpoint_path"],map_location="cpu")
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
            "fold_t": self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])


class TSNTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self.t_start_epoch = 0
        self.s_start_epoch = 0
        self.t_epoch_i = 0
        self.s_epoch_i = 0
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()

    def _build_optimizer(self):
        # optimizer
        self.t_embedder_optimizer = torch.optim.Adam(self.model.t_embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.t_encoder_optimizer = torch.optim.Adam(self.model.t_encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.t_decoder_optimizer = torch.optim.Adam(self.model.t_decoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.t_node_generater_optimizer = torch.optim.Adam(self.model.t_node_generater.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.t_merge_optimizer = torch.optim.Adam(self.model.t_merge.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        self.s_embedder_optimizer = torch.optim.Adam(self.model.s_embedder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_encoder_optimizer = torch.optim.Adam(self.model.s_encoder.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_decoder_optimizer1 = torch.optim.Adam(self.model.s_decoder_1.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_node_generater_optimizer1 = torch.optim.Adam(self.model.s_node_generater_1.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_merge_optimizer1 = torch.optim.Adam(self.model.s_merge_1.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_decoder_optimizer2 = torch.optim.Adam(self.model.s_decoder_2.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_node_generater_optimizer2 = torch.optim.Adam(self.model.s_node_generater_2.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.s_merge_optimizer2 = torch.optim.Adam(self.model.s_merge_2.parameters(), self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        # scheduler
        self.t_embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.t_embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.t_encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.t_encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.t_decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.t_decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.t_node_generater_scheduler = torch.optim.lr_scheduler.StepLR(self.t_node_generater_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.t_merge_scheduler = torch.optim.lr_scheduler.StepLR(self.t_merge_optimizer, step_size=self.config["step_size"], gamma=0.5)

        self.s_embedder_scheduler = torch.optim.lr_scheduler.StepLR(self.s_embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.s_encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.s_encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.s_decoder_scheduler1 = torch.optim.lr_scheduler.StepLR(self.s_decoder_optimizer1, step_size=self.config["step_size"], gamma=0.5)
        self.s_node_generater_scheduler1 = torch.optim.lr_scheduler.StepLR(self.s_node_generater_optimizer1, step_size=self.config["step_size"], gamma=0.5)
        self.s_merge_scheduler1 = torch.optim.lr_scheduler.StepLR(self.s_merge_optimizer1, step_size=self.config["step_size"], gamma=0.5)
        self.s_decoder_scheduler2 = torch.optim.lr_scheduler.StepLR(self.s_decoder_optimizer2, step_size=self.config["step_size"], gamma=0.5)
        self.s_node_generater_scheduler2 = torch.optim.lr_scheduler.StepLR(self.s_node_generater_optimizer2, step_size=self.config["step_size"], gamma=0.5)
        self.s_merge_scheduler2 = torch.optim.lr_scheduler.StepLR(self.s_merge_optimizer2, step_size=self.config["step_size"], gamma=0.5)

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "t_embedder_optimizer": self.t_embedder_optimizer.state_dict(),
            "t_encoder_optimizer": self.t_encoder_optimizer.state_dict(),
            "t_decoder_optimizer": self.t_decoder_optimizer.state_dict(),
            "t_generate_optimizer": self.t_node_generater_optimizer.state_dict(),
            "t_merge_optimizer": self.t_merge_optimizer.state_dict(),
            "t_embedder_scheduler": self.t_embedder_scheduler.state_dict(),
            "t_encoder_scheduler": self.t_encoder_scheduler.state_dict(),
            "t_decoder_scheduler": self.t_decoder_scheduler.state_dict(),
            "t_generate_scheduler": self.t_node_generater_scheduler.state_dict(),
            "t_merge_scheduler": self.t_merge_scheduler.state_dict(),
            "s_embedder_optimizer": self.s_embedder_optimizer.state_dict(),
            "s_encoder_optimizer": self.s_encoder_optimizer.state_dict(),
            "s_decoder_optimizer1": self.s_decoder_optimizer1.state_dict(),
            "s_generate_optimizer1": self.s_node_generater_optimizer1.state_dict(),
            "s_merge_optimizer1": self.s_merge_optimizer1.state_dict(),
            "s_decoder_optimizer2": self.s_decoder_optimizer2.state_dict(),
            "s_generate_optimizer2": self.s_node_generater_optimizer2.state_dict(),
            "s_merge_optimizer2": self.s_merge_optimizer2.state_dict(),
            "s_embedder_scheduler": self.s_embedder_scheduler.state_dict(),
            "s_encoder_scheduler": self.s_encoder_scheduler.state_dict(),
            "s_decoder_scheduler1": self.s_decoder_scheduler1.state_dict(),
            "s_generate_scheduler1": self.s_node_generater_scheduler1.state_dict(),
            "s_merge_scheduler1": self.s_merge_scheduler1.state_dict(),
            "s_decoder_scheduler2": self.s_decoder_scheduler2.state_dict(),
            "s_generate_scheduler2": self.s_node_generater_scheduler2.state_dict(),
            "s_merge_scheduler2": self.s_merge_scheduler2.state_dict(),
            "t_start_epoch": self.t_epoch_i,
            "s_start_epoch": self.s_epoch_i,
            "best_valid_value_accuracy": self.best_valid_value_accuracy,
            "best_valid_equ_accuracy": self.best_valid_equ_accuracy,
            "best_test_value_accuracy": self.best_test_value_accuracy,
            "best_test_equ_accuracy": self.best_test_equ_accuracy,
            "best_folds_accuracy": self.best_folds_accuracy,
            "fold_t": self.config["fold_t"]
        }
        torch.save(check_pnt, self.config["checkpoint_path"])

    def _load_checkpoint(self):
        return super()._load_checkpoint()

    def _teacher_net_train(self):
        self.model.t_embedder.train()
        self.model.t_encoder.train()
        self.model.t_decoder.train()
        self.model.t_node_generater.train()
        self.model.t_merge.train()
        self.model.s_embedder.eval()
        self.model.s_encoder.eval()
        self.model.s_decoder_1.eval()
        self.model.s_node_generater_1.eval()
        self.model.s_merge_1.eval()
        self.model.s_decoder_2.eval()
        self.model.s_node_generater_2.eval()
        self.model.s_merge_2.eval()

    def _student_net_train(self):
        self.model.t_embedder.eval()
        self.model.t_encoder.eval()
        self.model.t_decoder.eval()
        self.model.t_node_generater.eval()
        self.model.t_merge.eval()
        self.model.s_embedder.train()
        self.model.s_encoder.train()
        self.model.s_decoder_1.train()
        self.model.s_node_generater_1.train()
        self.model.s_merge_1.train()
        self.model.s_decoder_2.train()
        self.model.s_node_generater_2.train()
        self.model.s_merge_2.train()

    def _teacher_optimizer_step(self):
        self.t_embedder_optimizer.step()
        self.t_encoder_optimizer.step()
        self.t_decoder_optimizer.step()
        self.t_node_generater_optimizer.step()
        self.t_merge_optimizer.step()

    def _student_optimizer_step(self):
        self.s_embedder_optimizer.step()
        self.s_encoder_optimizer.step()
        self.s_decoder_optimizer1.step()
        self.s_node_generater_optimizer1.step()
        self.s_merge_optimizer1.step()
        self.s_decoder_optimizer2.step()
        self.s_node_generater_optimizer2.step()
        self.s_merge_optimizer2.step()

    def _teacher_scheduler_step(self):
        self.t_embedder_scheduler.step()
        self.t_encoder_scheduler.step()
        self.t_decoder_scheduler.step()
        self.t_node_generater_scheduler.step()
        self.t_merge_scheduler.step()

    def _student_scheduler_step(self):
        self.s_embedder_scheduler.step()
        self.s_encoder_scheduler.step()
        self.s_decoder_scheduler1.step()
        self.s_node_generater_scheduler1.step()
        self.s_merge_scheduler1.step()
        self.s_decoder_scheduler2.step()
        self.s_node_generater_scheduler2.step()
        self.s_merge_scheduler2.step()

    def _train_teacher_net_batch(self, batch):
        batch_loss = self.model.teacher_calculate_loss(batch)
        return batch_loss

    def _train_student_net_batch(self, batch):
        batch_loss = self.model.student_calculate_loss(batch)
        return batch_loss

    def _eval_teacher_net_batch(self, batch):
        test_out, target = self.model.teacher_test(batch)

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

    def _eval_student_net_batch(self, batch):
        test_out1, score1, test_out2, score2, target = self.model.student_test(batch)
        batch_size = len(test_out1)
        val_acc = []
        equ_acc = []
        s1_val_acc = []
        s1_equ_acc = []
        s2_val_acc = []
        s2_equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac1, equ_ac1, _, _ = self.evaluator.result(test_out1[idx], target[idx])
                val_ac2, equ_ac2, _, _ = self.evaluator.result(test_out2[idx], target[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac1, equ_ac1, _, _ = self.evaluator.result_multi(test_out1[idx], target[idx])
                val_ac2, equ_ac2, _, _ = self.evaluator.result_multi(test_out2[idx], target[idx])
            else:
                raise NotImplementedError
            if score1 > score2:
                val_acc.append(val_ac1)
                equ_acc.append(equ_ac1)
            else:
                val_acc.append(val_ac2)
                equ_acc.append(equ_ac2)
            s1_val_acc.append(val_ac1)
            s1_equ_acc.append(equ_ac1)
            s2_val_acc.append(val_ac2)
            s2_equ_acc.append(equ_ac2)
        return val_acc, equ_acc, s1_val_acc, s1_equ_acc, s2_val_acc, s2_equ_acc

    def _build_soft_target_batch(self, batch):
        self.model.init_soft_target(batch)

    def _train_epoch(self, module_name):
        epoch_start_time = time.time()
        loss_total = 0.
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            if module_name == 'teacher_net':
                self._teacher_net_train()
                batch_loss = self._train_teacher_net_batch(batch)
                self._teacher_optimizer_step()
            elif module_name == 'student_net':
                self._student_net_train()
                batch_loss = self._train_student_net_batch(batch)
                self._student_optimizer_step()
            else:
                NotImplementedError("TSN has no {} module".format(module_name))
            loss_total += batch_loss
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch_nums"]
        self.train_batch_nums = int(self.dataloader.trainset_nums / train_batch_size) + 1

        self.logger.info("start training...")

        self.logger.info("start training teacher net...")
        for epo in range(self.t_start_epoch, epoch_nums):
            self.t_epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch(module_name='teacher_net')
            self._teacher_scheduler_step()

            self.logger.info("epoch [%3d] teacher net avr loss [%2.8f] | train time %s"\
                                %(self.t_epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate_teacher(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac, valid_total, valid_time_cost = self.evaluate_teacher(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac, test_total, test_time_cost = self.evaluate_teacher(DatasetType.Test)

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

        self._load_model()
        self.logger.info("build soft target...")
        self.model.eval()
        for batch_idx, batch in enumerate(self.dataloader.load_data(DatasetType.Train)):
            self.batch_idx = batch_idx + 1
            self._build_soft_target_batch(batch)

        self.model.init_encoder_mask(self.config['train_batch_size'])
        self.logger.info("start training student net...")
        self.best_valid_value_accuracy = 0.
        self.best_valid_equ_accuracy = 0.
        self.best_test_value_accuracy = 0.
        self.best_test_equ_accuracy = 0.
        for epo in range(self.s_start_epoch, epoch_nums):
            self.s_epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch(module_name='student_net')
            self._student_scheduler_step()

            self.logger.info("epoch [%3d] student net avr loss [%2.8f] | train time %s"\
                                %(self.s_epoch_i,loss_total/self.train_batch_nums,train_time_cost))

            if epo % self.test_step == 0 or epo > epoch_nums - 5:
                if self.config["k_fold"]:
                    test_equ_ac, test_val_ac,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac, test_total, test_time_cost = self.evaluate_student(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | student1 equ acc [%2.3f] | student1 value acc [%2.3f] | student2 equ acc [%2.3f] | student2 value acc [%2.3f]"\
                                    %(test_total,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac))
                    self.logger.info("---------- test total [%d] | test equ acc [%2.3f] | test value acc [%2.3f] | test time %s"\
                                    %(test_total,test_equ_ac,test_val_ac,test_time_cost))

                    if test_val_ac >= self.best_test_value_accuracy:
                        self.best_test_value_accuracy = test_val_ac
                        self.best_test_equ_accuracy = test_equ_ac
                        self._save_model()
                else:
                    valid_equ_ac, valid_val_ac,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac, valid_total, valid_time_cost = self.evaluate_student(DatasetType.Valid)

                    self.logger.info("---------- valid total [%d] | student1 equ acc [%2.3f] | student1 value acc [%2.3f] | student2 equ acc [%2.3f] | student2 value acc [%2.3f]"\
                                    %(test_total,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac))
                    self.logger.info("---------- valid total [%d] | valid equ acc [%2.3f] | valid value acc [%2.3f] | valid time %s"\
                                    %(valid_total,valid_equ_ac,valid_val_ac,valid_time_cost))
                    test_equ_ac, test_val_ac,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac, test_total, test_time_cost = self.evaluate_student(DatasetType.Test)

                    self.logger.info("---------- test total [%d] | student1 equ acc [%2.3f] | student1 value acc [%2.3f] | student2 equ acc [%2.3f] | student2 value acc [%2.3f]"\
                                    %(test_total,s1_equ_ac,s1_val_ac,s2_equ_ac,s2_val_ac))
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

    def evaluate_teacher(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac = self._eval_teacher_net_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total, eval_total, test_time_cost

    def evaluate_student(self, eval_set):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        s1_value_ac = 0
        s1_equation_ac = 0
        s2_value_ac = 0
        s2_equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data(eval_set):
            batch_val_ac, batch_equ_ac, s1_val_ac, s1_equ_ac, s2_val_ac, s2_equ_ac = self._eval_student_net_batch(batch)
            value_ac += batch_val_ac.count(True)
            equation_ac += batch_equ_ac.count(True)
            s1_value_ac += s1_val_ac.count(True)
            s1_equation_ac += s1_equ_ac.count(True)
            s2_value_ac += s2_val_ac.count(True)
            s2_equation_ac += s2_equ_ac.count(True)
            eval_total += len(batch_val_ac)

        test_time_cost = time_since(time.time() - test_start_time)
        return equation_ac / eval_total, value_ac / eval_total,s1_equation_ac / eval_total,s1_value_ac / eval_total,\
                s2_equation_ac / eval_total,s2_value_ac / eval_total,eval_total, test_time_cost


class EPTTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()

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

    def _train_batch(self, batch):
        batch_loss = self.model.calculate_loss(batch)
        return batch_loss

    def _eval_batch(self, batch):
        '''seq, seq_length, group_nums, target'''
        test_out, target_out = self.model.model_test(batch)
        
        batch_size = len(test_out)
        val_acc = []
        equ_acc = []
        for idx in range(batch_size):
            if self.config["task_type"] == TaskType.SingleEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx], target_out[idx])
            elif self.config["task_type"] == TaskType.MultiEquation:
                val_ac, equ_ac, _, _ = self.evaluator.result_multi(test_out[idx], target_out[idx])
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

    def _build_optimizer(self):
        from torch_optimizer import Lamb
        self.optimizer = Lamb(self.model.parameters(), lr=self.config["learning_rate"], eps=1e-08)