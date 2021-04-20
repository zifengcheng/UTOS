import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class luong_gate_attention(nn.Module):

    def __init__(self, hidden_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.q = nn.Sequential(nn.Linear(hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        self.k = nn.Sequential(nn.Linear(2 * hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        self.v = nn.Sequential(nn.Linear(2 * hidden_size, 50, bias=False), nn.SELU(), nn.Dropout(p=0.2))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, h, context, time_step, win, mask=None):
        mask1 = torch.ones(h.shape[0],h.shape[1]).cuda()
        for i,j in enumerate(mask):
            if j < max(mask):
                mask1[mask[i]:,i]=0
        mask1 = mask1.permute(1,0)

        q = self.q(h)  # seq * batch * 50
        k = self.k(context)
        v = self.v(context)
        # print(q.shape,k.shape,v.shape)
        weights = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0)).squeeze(1)  # shape : batch * seq_len

        if mask is not None:
            weights = weights.masked_fill(mask1 == 0, -1e10)
        weights = self.softmax(weights / (q.shape[2] ** 0.5 ))
        # print(weights.shape,v.shape)
        c_t = torch.bmm(weights.unsqueeze(1), v.permute(1, 0, 2)).squeeze(1)
        # print( c_t.shape)
        return c_t, weights


class Decoder(torch.nn.Module):
    def __init__(self, encoder_hidden_size=100, decoder_hidden_size=100, attention_hidden_size=100, num_classes=10,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.label_embedding = torch.nn.Embedding(num_classes, 50)
        self.dropout = torch.nn.Dropout(dropout)

        # self.attention = Attention(encoder_hidden_size, decoder_hidden_size, attention_hidden_size)
        self.attention = luong_gate_attention(encoder_hidden_size)

        self.rnn = torch.nn.GRU(150, decoder_hidden_size)

        self.linear = torch.nn.Linear(2 * decoder_hidden_size + 50, decoder_hidden_size)
        self.hidden2label = torch.nn.Linear(decoder_hidden_size, num_classes)
        self.linear1 = torch.nn.Linear(2 * encoder_hidden_size, decoder_hidden_size)

    def forward(self, inputs, last_hidden, encoder_outputs, current_encoder_outputs, time_step, max_len, mask,
                inputs_mask=None):
        embedded = self.label_embedding(inputs).unsqueeze(0)
        #embedded = self.dropout(embedded)

        input = F.leaky_relu(self.linear1(current_encoder_outputs)).permute(1,0,2)
        #print('input shape:',input.shape)
        input1 = torch.cat((embedded, input), 2)
        output, hidden = self.rnn(input1, last_hidden)

        context, attn_weights = self.attention(output, encoder_outputs, time_step, 5, inputs_mask)
        context = context.unsqueeze(0)

        output = torch.cat([context, input, output], 2)
        output = F.leaky_relu(self.linear(output))
        output = self.hidden2label(output).squeeze(0)
        output = F.log_softmax(output, dim=1)

        return output, hidden

class Seq2Seq(torch.nn.Module):
    def __init__(self, decoder):
        super(Seq2Seq, self).__init__()
        self.gru = nn.GRU(768,100, bidirectional=True,batch_first=True)
        self.decoder = decoder
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, source, source_len, epoch, testing=False,target=None):
        # source : 32 * 75 * 30 , target : 32 * 75
        # now :  batch * seq * hidden_size
        output1 ,hidden = self.gru(source) 
        batch_size = source.size(0)
        max_len = max(source_len)  # in other sq2seq, max_len should be target.size()
        outputs = Variable(torch.zeros(max_len, batch_size,10)).cuda()

        #  encoder_outputs.shape   75 * 32 * 200
        hidden = hidden[:1]
        output = Variable(torch.zeros((batch_size))).long().cuda()
        mask = torch.zeros(batch_size, 10).long().cuda()
        for t in range(max_len):
            current_encoder_outputs = output1[:,t, :].unsqueeze(1)
            output, hidden = self.decoder(output, hidden, output1.permute(1,0,2), current_encoder_outputs, t,max_len, mask, source_len)
            outputs[t] = output
            is_teacher = random.random() < 1 - epoch * 0.05
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).cuda()
            elif is_teacher:
                target = torch.LongTensor(target)
                output = Variable(target.permute(1,0)[t]).cuda()
            else:
                output = Variable(top1).cuda()
        return outputs
