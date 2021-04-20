import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils.utils import *


torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader

def get_ecpair(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    return dataset.pair


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split

        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list = self.read_data_file(self.data_type)
        self.pair = self.get_pair(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]

        if bert_token_lens > 512:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens,
                                                                          doc_couples, y_emotions, y_causes, doc_len)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []

        data_list = read_json(data_file)
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            #print(type(doc['doc_len']))
            doc_couples = doc['pairs']
            doc_couples.sort()

            doc_emotions, doc_causes = zip(*doc_couples)
            #print(doc_emotions)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)

            y_emotions, y_causes = [], []
            doc_clauses = doc['clauses']
            #doc_str = '[CLS] '
            doc_str_1 = ''
            emotion_guided = True if len(list(set(doc_emotions))) != len(doc_emotions) else False
            cause_guided = True if len(list(set(doc_causes))) != len(doc_causes) else False
            emotion_index = list(set(doc_emotions))
            emotion_index.sort()
            cause_index = list(set(doc_causes))
            cause_index.sort()
            for i in range(doc_len):
                if cause_guided:
                    if i+1 in doc_emotions and i+1 in doc_causes:  # both 
                        emotion_label = cause_index.index(i+1)*3 + 3
                        cause_label = 0
                    elif i+1 in doc_emotions:   # emotion 
                        emotion_label = cause_index.index(doc_causes[doc_emotions.index(i+1)]) * 3 +1
                        cause_label = 0
                    elif i+1 in doc_causes:   # cause
                        emotion_label = cause_index.index(i+1) * 3 + 2
                        cause_label = 0
                    else:
                        emotion_label,cause_label = 0,0
                else:
                    if i+1 in doc_emotions and i+1 in doc_causes:  # both 
                        emotion_label = emotion_index.index(i+1)*3 + 3
                        cause_label = 0
                    elif i+1 in doc_emotions:   # emotion 
                        emotion_label = emotion_index.index(i+1)*3 +1
                        cause_label = 0
                    elif i+1 in doc_causes:   # cause
                        emotion_label = emotion_index.index(doc_emotions[doc_causes.index(i+1)]) * 3 + 2
                        cause_label = 0
                    else:
                        emotion_label,cause_label = 0,0
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)

                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                #doc_str += clause['clause'] + ' [SEP] '
                doc_str_1 +=' [CLS] '+ clause['clause'] + ' [SEP] '

            if int(doc_id) == 376:
                y_emotions = [0,0,0,0,0,0,2,1,0,0,5,4,0,0]
            #indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            indexed_tokens = self.bert_tokenizer.encode(doc_str_1.strip(), add_special_tokens=False)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            #segments_indices.insert(0,-1)
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)
            #print(indexed_tokens,segments_indices,segments_ids)
            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list

    def get_pair(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        data_list = read_json(data_file)
        pairs = []
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_couples.sort()
            pair_single = []
            for a in doc_couples:
                pair_single.extend([int(doc_id) * 10000 + a[0] * 100 + a[1]])
            pairs.extend([pair_single])
        return pairs

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_len = i
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(doc_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    #print(bert_segment_b.shape, bert_token_b.shape)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    return np.array(doc_len_b), np.array(adj_b), \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
           bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b


def pad_docs(doc_len_b, y_emotions_b, y_causes_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, 0)
        #print(y_emotions, max_doc_len)
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_causes_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_


def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad