import random

import numpy as np
import torch
from torch.utils.data import Dataset

def write_pred(test_pred,test_idx,file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path,'w') as f:
        for idx,pred in zip(test_idx,test_pred):
            print(idx+','+str(pred[0]),file=f)

# Dataset preparation
class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.benign_list = []
        for i in range(len(label_list)):
            if label_list[i] == 0:
                self.benign_list.append(i)
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        idx1 = random.choice(self.benign_list)
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                tlen = len(tmp)
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))
                if random.randint(1, 10) <= 5 and self.label_list[idx] == 1:
                    with open(self.data_path+self.fp_list[idx1], 'rb') as f1:
                        bs = f1.read(self.first_n_byte - tlen)
                        tmp[tlen:tlen+len(bs)] = [i + 1 for i in bs]
        except:
            with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                tlen = len(tmp)
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))
                if random.randint(1, 10) <= 5 and self.label_list[idx] == 1:
                   with open(self.data_path+self.fp_list[idx1], 'rb') as f1:
                        bs = f1.read(self.first_n_byte - tlen)
                        tmp[tlen:tlen+len(bs)] = [i + 1 for i in bs]

        return np.array(tmp),np.array([self.label_list[idx]])
