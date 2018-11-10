#!/usr/bin/env python

from torch.utils.data import Dataset, DataLoader
import numpy as np

def pad_data(s, length):
    padded = np.zeros((length,), dtype = np.int64)
    if len(s) > length: 
        padded = s[:length]
    else:
        padded[:len(s)] = s
    return np.array(padded)

def get_index(x, word2index):
    try:
        return word2index[x]
    except KeyError:
        return 1

class VectorizeData(Dataset):
    def __init__(self, df, word2index, label, maxlen=25):
        self.df = df
        self.label = label
        self.df.seq_len = [len(x.split(' ')) for x in self.df['clean_tweet']]
        self.df.numeric = [[get_index(y, word2index) for y in x.split(' ')] for x in self.df['clean_tweet']]
        self.df.padded_tweet = [pad_data(x, maxlen) for x in self.df.numeric]

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.padded_tweet[idx]
        y = self.df[self.label][idx]
        lens = self.df.seq_len[idx]
        return X,y,lens

