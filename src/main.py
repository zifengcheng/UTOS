import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from networks.rank_cp import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from pytorch_pretrained_bert.optimization import BertAdam
import random



def main(configs, fold_id):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] =str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
        valid_pair = get_ecpair(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    test_pair = get_ecpair(configs, fold_id=fold_id, data_type='test')
    model = Network(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.seq.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-5},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0},
        {'params': params_rest,'lr':0.0005,'weight_decay': 1e-7}
    ]
    optimizer = BertAdam(params,
                             lr=1e-5,
                             warmup=0.1,
                             t_total=len(train_loader) // configs.gradient_accumulation_steps * configs.epochs)



    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    max_ec, max_e, max_c = (-1, -1, -1), None, None
    metric_ec, metric_e, metric_c = (-1,-1,-1), None, None
    early_stop_flag = None
    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
            pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b,epoch,False,y_emotions_b)  # seq_len * batch * 10
            loss = model.loss_pre(pred_e, y_emotions_b, doc_len_b)
            loss = loss / configs.gradient_accumulation_steps
            if train_step <= 20:
                print('epoch: ',epoch,loss)

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                #nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            show_time = 200
            if train_step % show_time ==0:
                with torch.no_grad():
                    model.eval()
                    if configs.split == 'split10':
                        test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
                        if test_ec[2] > metric_ec[2]:
                            early_stop_flag = 1
                            metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                            print('epoch',epoch,'test  p,r,f1 is :',test_ec)
                        else:
                            early_stop_flag += 1

                    if configs.split == 'split20':
                        valid_ec, valid_e, valid_c, _, _, _ = inference_one_epoch(configs, valid_loader, model,valid_pair,1)
                        test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
                        if valid_ec[2] > max_ec[2]:
                            print('epoch',epoch,'valid  p,r,f1 is :',valid_ec)
                            print('epoch',epoch,'test  p,r,f1 is :',test_ec)
                            early_stop_flag = 1
                            max_ec = valid_ec
                            metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                        else:
                            early_stop_flag += 1
                    
        with torch.no_grad():
            model.eval()
            if configs.split == 'split10':
                test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
                if test_ec[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                    print('epoch',epoch,'test  p,r,f1 is :',test_ec)
                else:
                    early_stop_flag += 1

            if configs.split == 'split20':
                valid_ec, valid_e, valid_c, _, _, _ = inference_one_epoch(configs, valid_loader, model,valid_pair,1)
                test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model,test_pair,1)
                if valid_ec[2] > max_ec[2]:
                    print('epoch',epoch,'valid  p,r,f1 is :',valid_ec)
                    print('epoch',epoch,'test  p,r,f1 is :',test_ec)
                    early_stop_flag = 1
                    max_ec = valid_ec
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break
        #print('epoch',epoch,'is finish')
    return metric_ec, metric_e, metric_c


def inference_one_batch(configs, batch, model,epoch):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
    pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,bert_clause_b, doc_len_b,epoch,True)
    doc_couples_pred_b = inference_pair(pred_e,doc_id_b)
    return doc_couples_b, doc_couples_pred_b, doc_id_b


def inference_one_epoch(configs, batches, model,pair,epoch):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    for batch in batches:
        doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(configs, batch, model,epoch)
        doc_id_all.extend(doc_id_b)
        doc_couples_pred_all.extend([doc_couples_pred])

    metric_ec, metric_e, metric_c = eval_func(pair, doc_couples_pred_all)
    return metric_ec, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all

def inference_pair( predict,doc_id_b):
    pre_test_1, pre_test_2, pre_test_3 = [], [], []

    result = predict.argmax(dim=2).permute(1,0).cpu().numpy()  # batch *seq_len

    for i in range(result.shape[0]):
        e1, e2, e3, c1, c2, c3 = [], [], [], [], [], []
        for j in range(result.shape[1]):
            if result[i][j] != 0:
                if int((result[i][j] + 2) / 3) == 1:
                    if result[i][j] % 3 == 1:
                        e1.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c1.append(j + 1)
                    else:
                        e1.append(j + 1)
                        c1.append(j + 1)
                elif int((result[i][j] + 2) / 3) == 2:
                    if result[i][j] % 3 == 1:
                        e2.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c2.append(j + 1)
                    else:
                        e2.append(j + 1)
                        c2.append(j + 1)
                else:
                    if result[i][j] % 3 == 1:
                        e3.append(j + 1)
                    elif result[i][j] % 3 == 2:
                        c3.append(j + 1)
                    else:
                        e3.append(j + 1)
                        c3.append(j + 1)

        for p in e1:
            for q in c1:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
        for p in e2:
            for q in c2:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
        for p in e3:
            for q in c3:
                pre_test_1.append(int(doc_id_b[i])*10000 + p * 100 + q)
    #if pre_test_1 != None:
        #print(pre_test_1)
    return pre_test_1

if __name__ == '__main__':
    configs = Config()

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 20
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    for fold_id in range(1, n_folds+1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp: {}'.format(metric_ec))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c}, '{}_{}_metrics.pkl'.format(time.time(), configs.split))

