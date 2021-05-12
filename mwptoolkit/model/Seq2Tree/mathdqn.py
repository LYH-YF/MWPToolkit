from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from mwptoolkit.module.Embedder.basic_embedder import BaiscEmbedder
from mwptoolkit.module.Encoder.rnn_encoder import BasicRNNEncoder,SelfAttentionRNNEncoder
from mwptoolkit.module.Layer.tree_layers import DQN,BinaryTree
from mwptoolkit.module.Environment.env import Env
from mwptoolkit.utils.enum_type import SpecialTokens,NumMask
from mwptoolkit.utils.data_structure import GoldTree

class MathDQN(nn.Module):
    def __init__(self,config):
        super(MathDQN,self).__init__()
        self.out_idx2symbol=config['out_idx2symbol']
        self.generate_list=config['generate_list']
        self.num_start = config['num_start']
        self.operator_list = config["operator_list"]
        self.replay_size=config['replay_size']
        self.max_out_len=30
        self.embedder=BaiscEmbedder(config['vocab_size'],config['embedding_size'],config['dropout_ratio'],padding_idx=0)
        # self.encoder=BasicRNNEncoder(config['embedding_size'],config['hidden_size'],config['num_layers'],\
        #                                 config['rnn_cell_type'],config['dropout_ratio'],config['bidirectional'])
        self.encoder=SelfAttentionRNNEncoder(config['embedding_size'],config['hidden_size'],config['embedding_size'],config['num_layers'],\
                                        config['rnn_cell_type'],config['dropout_ratio'],config['bidirectional'])
        self.dqn=DQN(config['embedding_size']*2,config['embedding_size']*2,config['hidden_size'],config['operator_nums'],config['dropout_ratio'])
        self.env=Env()
    def forward(self,seq,seq_length,num_pos,num_list,ans,target=None):
        batch_size=seq.size(0)
        device=seq.device

        seq_emb=self.embedder(seq)
        encoder_output, encoder_hidden = self.encoder(seq_emb, seq_length)
        
        generate_num=[self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)]+[self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num=torch.tensor(generate_num).to(device)
        generate_emb=self.embedder(generate_num)
        tree=[]
        look_ups=[]
        embs=[]
        for b_i in range(batch_size):
            tree.append(self.equ2tree(target[b_i],num_list[b_i],ans[b_i]))
            look_up = [SpecialTokens.UNK_TOKEN]+self.generate_list + num_list[b_i]
            num_embedding=torch.cat([generate_emb,encoder_output[b_i,num_pos[b_i]]],dim=0)
            num_list_, emb_ = self.get_num_list(target[b_i],num_list[b_i],look_up,num_embedding)
            look_ups.append(num_list_)
            embs.append(emb_)
        self.env.make_env(tree,look_ups,embs,self.operator_list)
        self.replay_memory=deque(maxlen=self.replay_size)
        for b_i in range(batch_size):
            obs=self.env.reset()
            for step in range(self.max_out_len):
                obs = obs.to(device)
                action,next_obs=self.dqn.play_one(obs)
                n_o, reward, done=self.env.step(action)
                if n_o != None:
                    next_obs=n_o
                self.replay_memory.append((obs, action, reward, next_obs, done))
                obs=next_obs
                if done:
                    break
        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
        dones = dones.to(device)
        rewards = rewards.to(device)
        self.dqn.eval()
        next_Q_values,_ = self.dqn(next_states)
        self.dqn.train()
        max_next_Q_values = torch.max(next_Q_values, dim=1)[0]
        discount_rate=0.95
        target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values

        mask = torch.zeros(batch_size, len(self.operator_list))
        idxs = torch.arange(0,batch_size)
        mask[idxs,actions]=1
        mask = mask.to(device)
        
        all_Q_values,_ = self.dqn(states)
        Q_values = torch.sum(all_Q_values * mask,dim=1)
        return Q_values,target_Q_values
    def predict(self,seq,seq_length,num_pos,num_list,ans,target=None):
        batch_size=seq.size(0)
        device=seq.device

        seq_emb=self.embedder(seq)
        encoder_output, encoder_hidden = self.encoder(seq_emb, seq_length)
        
        generate_num=[self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)]+[self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num=torch.tensor(generate_num).to(device)
        generate_emb=self.embedder(generate_num)
        tree=[]
        look_ups=[]
        embs=[]
        for b_i in range(batch_size):
            tree.append(self.equ2tree(target[b_i],num_list[b_i],ans[b_i]))
            look_up = [SpecialTokens.UNK_TOKEN]+self.generate_list + num_list[b_i]
            num_embedding=torch.cat([generate_emb,encoder_output[b_i,num_pos[b_i]]],dim=0)
            num_list_, emb_ = self.get_num_list(target[b_i],num_list[b_i],look_up,num_embedding)
            look_ups.append(num_list_)
            embs.append(emb_)
        self.env.make_env(tree,look_ups,embs,self.operator_list)
        acc=0
        for b_i in range(batch_size):
            obs=self.env.validate_reset(b_i)
            for step in range(self.max_out_len):
                obs = obs.to(device)
                action,next_obs=self.dqn.play_one(obs)
                n_o, done, flag=self.env.val_step(action)
                if n_o != None:
                    next_obs=n_o
                obs=next_obs
                if done:
                    if flag:
                        acc+=1
                    break
        return acc


    def sample_experiences(self,batch_size):
        indices = torch.randint(len(self.replay_memory), size=(batch_size,1))
        batch = [self.replay_memory[index] for index in indices]
        
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        
        for experience in batch:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            dones.append(experience[4])
        states=torch.stack(states)
        actions=torch.cat(actions)
        next_states=torch.stack(next_states)
        dones=torch.tensor(dones)
        rewards=torch.tensor(rewards)
        return states, actions, rewards, next_states, dones
    
    def equ2tree(self,equation,num_list,ans):
        tree=GoldTree()
        tree.equ2tree(equation,self.out_idx2symbol,self.operator_list,num_list,ans)
        return tree
    def get_num_list(self,equation,num_list,look_up,emb):
        num_list_=[]
        emb_list=[]
        for idx in equation:
            if idx>self.num_start:
                symbol=self.out_idx2symbol[idx]
                if symbol in NumMask.number:
                    i=NumMask.number.index(symbol)
                    num=num_list[i]
                else:
                    num=symbol
                if num in num_list_:
                    continue
                i=look_up.index(num)
                emb_list.append(emb[i])
                num_list_.append(num)
        return num_list_,emb_list
