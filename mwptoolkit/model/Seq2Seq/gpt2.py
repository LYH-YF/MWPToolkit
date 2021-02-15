import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer

from mwptoolkit.utils.enum_type import SpecialTokens,NumMask

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        #self.eval_generate_num = config['eval_generate_num']
        self.device=config["device"]
        self.pretrained_model_path = config['pretrained_model_path']
        
        self.tokenizer=BertTokenizer.from_pretrained(self.pretrained_model_path)
        _ =self.tokenizer.add_tokens(NumMask.alphabet)
        self.tokenizer.add_special_tokens({"additional_special_tokens":["<ans>"]})
        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)

        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.max_out_len = config['max_output_len']
    def forward(self, seq,target=None):
        
        if target != None:
            token_logits,target=self.generate_t(seq,target)
            return token_logits,target
        else:
            all_output=self.generate_without_t(seq)
            return all_output,None
    def list2str(self,x):
        y=''.join(x)
        return y
    def generate_t(self,seq,target=None):
        start_idx=[]
        output_len=0
        target_idx=[]
        batch_size=len(seq)
        for idx,s in enumerate(seq):
            seq[idx]=self.tokenizer.tokenize(seq[idx])
            t=target[idx]
            target_len=len(t)
            if output_len<target_len:
                output_len=target_len
            start_idx.append(len(seq[idx])+1)
            seq[idx]+=(["<ans>"]+t)
        encoding_dict=self.tokenizer.batch_encode_plus(seq,
                                            max_length=128,
                                            pad_to_max_length=True)
        input_ids=encoding_dict['input_ids']
        attn_masks=encoding_dict['attention_mask']
        for b in range(len(start_idx)):
            target_idx.append(input_ids[b][start_idx[b]:start_idx[b]+output_len])
        target_idx=torch.tensor(target_idx).to(self.device)
        input_ids=torch.tensor(input_ids).long().to(self.device)
        attn_masks=torch.tensor(attn_masks).bool().to(self.device)
        
        outputs = self.decoder(input_ids,
                                attention_mask=attn_masks)
        token_logits=[]
        for b in range(batch_size):
            token_logits.append(outputs[0][b,start_idx[b]:start_idx[b]+output_len,:])
        token_logits=torch.stack(token_logits,dim=0)
        token_logits=token_logits.view(-1,token_logits.size(-1))
        return token_logits,target_idx
    def generate_without_t(self,seq):
        all_output=[]
        for idx,s in enumerate(seq):
            seq[idx]=self.tokenizer.tokenize(seq[idx])
            seq[idx]+=["<ans>"]
        encoding_dict=self.tokenizer.batch_encode_plus(seq)
        input_ids=encoding_dict['input_ids']
        attn_masks=encoding_dict['attention_mask']
        input_ids=torch.tensor(input_ids).long().to(self.device)
        attn_masks=torch.tensor(attn_masks).bool().to(self.device)
        for idx in range(self.max_out_len):
            outputs = self.decoder(input_ids,
                                attention_mask=attn_masks)
            token_logit=outputs[0][:,-1,:]
            tokens=token_logit.topk(1,dim=1)[1]
            mask=tokens==self.tokenizer.pad_token_id
            all_output.append(tokens)
            input_ids=torch.cat((input_ids,tokens),dim=1)
            attn_masks=torch.cat((attn_masks,mask),dim=1)
        all_output=torch.cat(all_output,dim=1)
        all_output=self.decode(all_output)
        return all_output
    def decode(self,outputs):
        batch_size=outputs.size(0)
        all_outputs=[]
        for b in range(batch_size):
            symbols=self.tokenizer.decode(outputs[b])
            symbols=self.tokenizer.tokenize(symbols)
            all_outputs.append(symbols)
        return all_outputs



# if __name__ == '__main__':
#     config={
#         "pretrained_model_path":r"C:\Users\74765\.vscode\Programs\MWPToolkit\pretrain\gpt2",
#         "out_sos_token":1,
#         "out_eos_token":2,
#         "max_out_len":20
#     }
#     model=GPT2(config)
#     seq=["5（3）班的同学在母亲节都表达了对妈妈的节日祝福．其中，NUM的同学送了鲜花，NUM的同学给了妈妈一个香香的吻，其余的同学都送上了自制的贺卡．送自制贺卡的同学占全班的几分之几？"]
#     seq=[
#         "5 （ 3 ） 班 的 同学 在 母亲节 都 表达 了 对 妈妈 的 节日 祝福 ． 其中 ， NUM_a 的 同学 送 了 鲜花 ， NUM_b 的 同学 给 了 妈妈 一个 香香的 吻 ， 其余 的 同学 都 送 上 了 自制 的 贺卡 ． 送 自制 贺卡 的 同学 占 全班 的 几分 之 几 ？",
#         #"+ - * / ^",
#     ]
#     target=[["NUM_a","+","NUM_b"],["NUM_a"]]
#     #seq[0]=list(seq[0])
#     #seq[0]=seq[0].split(" ")
#     model.calculate_loss(seq)
#     print(1)