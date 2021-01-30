import torch
import time
from mwptoolkit.utils.utils import *
from mwptoolkit.utils.enum_type import PAD_TOKEN
from mwptoolkit.loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from mwptoolkit.loss.nll_loss import NLLLoss
from mwptoolkit.model.Seq2Tree.gts import GTS
from mwptoolkit.model.Seq2Seq.rnnencdec import RNNEncDec

class AbstractTrainer(object):
    def __init__(self,config,model,dataloader,evaluator):
        super().__init__()
        self.config=config
        self.model=model
        self.dataloader=dataloader
        self.evaluator=evaluator
        self.best_equ_accuracy=0.
        self.best_value_accuracy=0.
        self.start_epoch=0
        self.epoch_i=0
    def _save_checkpoint(self):
        raise NotImplementedError
    def _load_checkpoint(self):
        raise NotImplementedError
    def _save_model(self):
        state_dict={"model":self.model.state_dict()}
        torch.save(state_dict,self.config["trained_model_path"])
    def _load_model(self):
        state_dict=torch.load(self.config["trained_model_path"])
        self.model.load_state_dict(state_dict["model"])
    def _build_optimizer(self):
        raise NotImplementedError
    def fit(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError

class SingleEquationTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self._build_loss(config["symbol_size"],
                            self.dataloader.dataset.out_symbol2idx[PAD_TOKEN])
    
    def _build_optimizer(self):
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.config["learning_rate"])
    
    def _save_checkpoint(self):
        check_pnt = {
            "model":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "value_acc": self.best_value_accuracy,
            "equ_acc": self.best_equ_accuracy
        }
        torch.save(check_pnt, self.config["checkpoint_path"])
    
    def _load_checkpoint(self):
        check_pnt = torch.load(self.config["checkpoint_path"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_value_accuracy = check_pnt["value_acc"]
        self.best_equ_accuracy = check_pnt["equ_acc"]
    
    def _build_loss(self,symbol_size,out_pad_token):
        weight=torch.ones(symbol_size).to(self.config["device"])
        pad=out_pad_token
        self.loss=NLLLoss(weight,pad)
    def _idx2word_2idx(self,batch_equation):
        batch_size,length=batch_equation.size()
        batch_equation_=[]
        for b in range(batch_size):
            equation=[]
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[\
                                            self.dataloader.dataset.in_idx2word[\
                                                batch_equation[b,idx]]])
            batch_equation_.append(equation)
        batch_equation_=torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_
    def _train_batch(self,batch):
        outputs=self.model(batch["question"],batch["ques len"],batch["equation"])
        outputs=torch.nn.functional.log_softmax(outputs,dim=1)
        if self.config["share_vocab"]:
            batch_equation=self._idx2word_2idx(batch["equation"])
            self.loss.eval_batch(outputs,batch_equation.view(-1))
        else:
            self.loss.eval_batch(outputs,batch["equation"].view(-1))
        batch_loss = self.loss.get_loss()
        return batch_loss
    
    def _eval_batch(self,batch):
        test_out=self.model(batch["question"],batch["ques len"])
        if self.config["share_vocab"]:
            target=self._idx2word_2idx(batch["equation"])
        else:
            target=batch["equation"]
        batch_size=target.size(0)
        val_acc=[]
        equ_acc=[]
        for idx in range(batch_size):
            val_ac, equ_ac, _, _ = self.evaluator.result(test_out[idx],target[idx],batch["num list"][idx])
            val_acc.append(val_ac)
            equ_acc.append(equ_ac)
        return val_acc,equ_acc
    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self.model.train()
        for batch_idx, batch in enumerate(
                self.dataloader.load_data("train")):
            self.batch_idx = batch_idx + 1
            self.model.zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self.optimizer.step()
            self.loss.reset()
        epoch_time_cost=time_since(time.time() -epoch_start_time)
        return loss_total,epoch_time_cost
    
    def fit(self):
        train_batch_size=self.config["train_batch_size"]
        epoch_nums=self.config["epoch_nums"]
        
        self.train_batch_nums = int(
            self.dataloader.trainset_nums / train_batch_size) + 1
        
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total,train_time_cost=self._train_epoch()
            print("epoch [%2d] avr loss [%2.8f]"%(self.epoch_i,loss_total/self.train_batch_nums))
            print("---------- train time {}".format(train_time_cost))
            if epo % 2 == 0 or epo > epoch_nums - 5:
                equation_ac,value_ac,eval_total,test_time_cost=self.evaluate()
                print("---------- test equ acc [%2.3f] | test value acc [%2.3f]".format(equation_ac,value_ac))
                print("---------- test time {}".format(test_time_cost))
                if value_ac>=self.best_value_accuracy:
                    self.best_value_accuracy=value_ac
                    self.best_equ_accuracy=equation_ac
                    self._save_model()
            if epo%5==0:
                self._save_checkpoint()
    def evaluate(self):
        self.model.eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data("test"):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            value_ac+=batch_val_ac.count(True)
            equation_ac+=batch_equ_ac.count(True)
            eval_total+=len(batch_val_ac)
            # value_ac += batch_val_ac.count(True)
            # equation_ac += batch_equ_ac.count(True)
            # eval_total += len(batch_val_ac)
        test_time_cost=time_since(time.time()-test_start_time)
        return equation_ac/eval_total,value_ac/eval_total,eval_total,test_time_cost

class GTSTrainer(AbstractTrainer):
    def __init__(self, config, model, dataloader, evaluator):
        super().__init__(config, model, dataloader,evaluator)
        self._build_optimizer()
        if config["resume"]:
            self._load_checkpoint()
        self.loss=MaskedCrossEntropyLoss()
    def _build_optimizer(self):
        # optimizer
        self.embedder_optimizer=torch.optim.Adam(self.model.embedder.parameters(),
                                                    self.config["learning_rate"],
                                                    weight_decay=self.config["weight_decay"])
        self.encoder_optimizer=torch.optim.Adam(self.model.encoder.parameters(),
                                                    self.config["learning_rate"],
                                                    weight_decay=self.config["weight_decay"])
        self.decoder_optimizer=torch.optim.Adam(self.model.parameters(),
                                                    self.config["learning_rate"],
                                                    weight_decay=self.config["weight_decay"])
        self.node_generater_optimizer=torch.optim.Adam(self.model.node_generater.parameters(),
                                                    self.config["learning_rate"],
                                                    weight_decay=self.config["weight_decay"])
        self.merge_optimizer=torch.optim.Adam(self.model.merge.parameters(),
                                                    self.config["learning_rate"],
                                                    weight_decay=self.config["weight_decay"])
        # scheduler
        self.embedder_scheduler = torch.optim.lr_scheduler.StepLR(
            self.embedder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(
            self.decoder_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.node_generater_scheduler = torch.optim.lr_scheduler.StepLR(
            self.node_generater_optimizer, step_size=self.config["step_size"], gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(
            self.merge_optimizer, step_size=self.config["step_size"], gamma=0.5)
    
    def _save_checkpoint(self):
        check_pnt = {
            "model":self.model.state_dict(),
            "embedder_optimizer": self.embedder_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_optimizer": self.node_generater_optimizer.state_dict(),
            "merge_optimizer": self.merge_optimizer.state_dict(),
            "embedder_scheduler": self.embedder_scheduler.state_dict(),
            "encoder_scheduler": self.encoder_scheduler.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "generate_scheduler": self.node_generater_scheduler.state_dict(),
            "merge_scheduler": self.merge_scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "value_acc": self.best_value_accuracy,
            "equ_acc": self.best_equ_accuracy
        }
        torch.save(check_pnt, self.config["checkpoint_path"])
    def _load_checkpoint(self):    
        check_pnt = torch.load(self.config["checkpoint_path"])
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.embedder_optimizer.load_state_dict(
            check_pnt["embedder_optimizer"])
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.decoder_optimizer.load_state_dict(check_pnt["decoder_optimizer"])
        self.node_generater_optimizer.load_state_dict(
            check_pnt["generate_optimizer"])
        self.merge_optimizer.load_state_dict(check_pnt["merge_optimizer"])
        #load parameter of scheduler
        self.embedder_scheduler.load_state_dict(
            check_pnt["embedding_scheduler"])
        self.encoder_scheduler.load_state_dict(check_pnt["encoder_scheduler"])
        self.decoder_scheduler.load_state_dict(check_pnt["decoder_scheduler"])
        self.node_generater_scheduler.load_state_dict(
            check_pnt["generate_scheduler"])
        self.merge_scheduler.load_state_dict(check_pnt["merge_scheduler"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_value_accuracy = check_pnt["value_acc"]
        self.best_equ_accuracy = check_pnt["equ_acc"]
    
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

    def _train_batch(self,batch):
        '''
        seq, seq_length, nums_stack, num_size, generate_nums, num_pos,\
                UNK_TOKEN,num_start,target=None, target_length=None,max_length=30,beam_size=5
        '''
        unk = self.dataloader.out_unk_token
        num_start = self.dataloader.dataset.num_start
        generate_nums=[self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]

        outputs=self.model(batch["question"],batch["ques len"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start,batch["equation"],batch["equ len"],UNK_TOKEN=unk)
        self.loss.eval_batch(outputs,batch["equation"],batch["equ mask"])
        batch_loss = self.loss.get_loss()
        return batch_loss
    def _eval_batch(self,batch):
        num_start = self.dataloader.dataset.num_start
        generate_nums=[self.dataloader.dataset.out_symbol2idx[symbol] for symbol in self.dataloader.dataset.generate_list]
        test_out=self.model(batch["question"],batch["ques len"],batch["num stack"],batch["num size"],\
                                generate_nums,batch["num pos"],num_start)
        
        val_ac, equ_ac, _, _ = self.evaluator.prefix_tree_result(
                test_out, batch["equation"].tolist()[0],batch["num list"][0], batch["num stack"][0])
        return val_ac,equ_ac
        
    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = 0.
        self._model_train()
        for batch_idx, batch in enumerate(
                self.dataloader.load_data("train")):
            self.batch_idx = batch_idx + 1
            self._model_zero_grad()
            batch_loss = self._train_batch(batch)
            loss_total += batch_loss
            self.loss.backward()
            self._optimizer_step()
            self.loss.reset()
        epoch_time_cost=time_since(time.time() -epoch_start_time)
        return loss_total,epoch_time_cost
        #print("epoch [%2d]avr loss [%2.8f]"%(self.epoch_i,loss_total /self.batch_nums))
        #print("epoch train time {}".format(time_since(time.time() -epoch_start_time)))
    
    def fit(self):
        train_batch_size=self.config["train_batch_size"]
        epoch_nums=self.config["epoch_nums"]
        self.train_batch_nums = int(
            self.dataloader.trainset_nums / train_batch_size) + 1
        for epo in range(self.start_epoch, epoch_nums):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total,train_time_cost=self._train_epoch()
            self._scheduler_step()
            print("epoch [%2d] avr loss [%2.8f]"%(self.epoch_i,loss_total/self.train_batch_nums))
            print("---------- train time {}".format(train_time_cost))
            if epo % 10 == 0 or epo > epoch_nums - 5:
                equation_ac,value_ac,eval_total,test_time_cost=self.evaluate()
                print("---------- test equ acc [%2.3f] | test value acc [%2.3f]".format(equation_ac,value_ac))
                print("---------- test time {}".format(test_time_cost))
                if value_ac>=self.best_value_accuracy:
                    self.best_value_accuracy=value_ac
                    self.best_equ_accuracy=equation_ac
                    self._save_model()
            if epo%5==0:
                self._save_checkpoint()
    
    def evaluate(self):
        self._model_eval()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        test_start_time = time.time()
        for batch in self.dataloader.load_data("test"):
            batch_val_ac, batch_equ_ac = self._eval_batch(batch)
            if batch_val_ac:
                value_ac+=1
            if batch_equ_ac:
                equation_ac+=1
            eval_total+=1
            # value_ac += batch_val_ac.count(True)
            # equation_ac += batch_equ_ac.count(True)
            # eval_total += len(batch_val_ac)
        test_time_cost=time_since(time.time()-test_start_time)
        return equation_ac/eval_total,value_ac/eval_total,eval_total,test_time_cost
