import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from networks.our_model import *
from pytorch_pretrained_bert import BertModel

class BertEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_cache_path)
    
    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
        
    def forward(self, document_list,document_len,bert_segment_b):
        text_list, tokens_list, ids_list = [], [], []
        #print(document_len)
        c = document_list.numpy().tolist()
        s = bert_segment_b.numpy().tolist()
        ids_list = []
        seg_list = []
        for i,k in zip(c,s) :
            cc,start =[], 0
            sg = []
            for j in range(i.count(102)):
                start1 = i.index(102,start+1)
                cc.append(i[start:start1+1])
                sg.append(k[start:start1+1])
                start =  start1 + 1
            ids_list.extend(cc)
            seg_list.extend(sg)
   
        ids_padding_list, mask_list = self.padding_and_mask(ids_list)
        ids_padding_tensor = torch.LongTensor(ids_padding_list).cuda()

        mask_tensor = torch.tensor(mask_list).cuda()
        
        bert_segment_b,_ = self.padding_and_mask(seg_list)
        bert_segment_b = torch.LongTensor(bert_segment_b).cuda()

        #print(ids_padding_tensor.shape,mask_tensor.shape,bert_segment_b.shape)

        _, pooled = self.bert(ids_padding_tensor, attention_mask = mask_tensor, token_type_ids=bert_segment_b.cuda(),output_all_encoded_layers=False)
        
        start = 0
        clause_state_list = []
        for dl in document_len:
            end = start + dl
            if pooled[start: end].shape[0] == max(document_len):
                clause_state_list.append(pooled[start: end])
            else:
                padding1 = torch.zeros(max(document_len) - pooled[start: end].shape[0] ,pooled[start: end].shape[1]).cuda()
                clause_state_list.append(torch.cat((pooled[start: end],padding1),0))
            start = end
        clause = torch.stack(clause_state_list,0)
        return pooled, clause


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertEncoder(configs)
        self.decoder = Decoder()
        self.seq = Seq2Seq(self.decoder)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, epoch,testing=False,target=None):

        pooled , bert_output = self.bert(bert_token_b,doc_len,bert_segment_b)
        pred_e = self.seq(bert_output, doc_len, epoch,testing,target)

        return pred_e


    def loss_pre(self, pred_e, y_emotions, source_length):
        #print('loss function shape is ',pred_e.shape,y_emotions.shape)   #seq_len * batch  * class  .  batch * seq_len
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_e, list(source_length),enforce_sorted=False).data
        target_ = torch.nn.utils.rnn.pack_padded_sequence(y_emotions.permute(1,0), list(source_length),enforce_sorted=False).data
        loss_e  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)

        return loss_e

