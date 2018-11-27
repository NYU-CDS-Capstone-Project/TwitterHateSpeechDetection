#!/usr/bin/python

"""
python train.py label dataset weights_matrix_id vocab_size seq_len
label: name of the label you want to predict
(e.g. 'CAPS', 'Obscenity', 'Threat', 'hatespeech', 'namecalling', 'negprejudice', 'noneng', 'porn', 'stereotypes')
dataset: filename of the dataset (e.g. train_nn.csv)
weights matrix: id in the filename of the weights matrix (e.g. 1)
vocab_size: size of vocabulary (e.g. 10000) must be consistent with weight matrix.
seq_len: maximum length of sequences before truncation (e.g. 25)
"""

import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import model_selection
from utils import *
from model import *
from data_loader import *
import pickle
import sys
np.random.seed(0)

#drew inspiration from
#https://github.com/dmesquita/understanding_pytorch_nn and
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
#https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch.ipynb
#https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
#https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb

print ('label to predict: ', sys.argv[1])
print ('data file: ', sys.argv[2])

label = sys.argv[1]
filename = sys.argv[2]
weights_matrix_id = sys.argv[3]
vocab_size = int(sys.argv[4])
seq_len = int(sys.argv[5])

train = pd.read_csv("../" + str(filename))

train = train[train.clean_tweet.isnull() == False]

train_sub, validation = model_selection.train_test_split(train, test_size = 0.2, random_state = 123)
train_sub.reset_index(inplace = True, drop = True)
validation.reset_index(inplace = True, drop = True)

vocab = build_vocab(vocab_size, train_sub.clean_tweet)
word2index, index2word = build_idx(vocab)

weights_matrix_filename = "weights_matrix_" + str(weights_matrix_id) + ".sav"

weights = open(weights_matrix_filename,"rb")
weights = pickle.load(weights)

val = VectorizeData(validation, word2index, label = label, maxlen = seq_len)

hidden_size = 200
learning_rate = 0.001
batch_size = 32

best_net = LSTMClassifier(weights, weights.shape[0], hidden_size, hidden_size, 2, batch_size)
best_net.load_state_dict(torch.load('best_rnn_' + label + '.pt'))
best_net.eval()

torch.manual_seed(0)
dl_val = DataLoader(val, batch_size = 32, shuffle = False, num_workers = 0, drop_last = True)

final_predictions, val_labels = get_validation_predictions(dl_val, best_net)

print ("confusion matrix " + str(confusion_matrix(val_labels, final_predictions)))

print ("f-score of best_net " + str(f1_score(final_predictions, val_labels)))

validation_subset = validation[validation[label + '_pct'].isin([0, 1])]
validation_subset.reset_index(inplace = True, drop = True)

val_subset = VectorizeData(validation_subset, word2index, label = label, maxlen = seq_len)
torch.manual_seed(0)
dl3 = DataLoader(val_subset, batch_size = 32, shuffle = False, num_workers = 0, drop_last = True)

prediction_subset, label_subset = get_validation_predictions(dl3, best_net)

print ("confusion matrix with agreement" + str(confusion_matrix(label_subset, prediction_subset)))

print ("f-score of best_net on samples with agreement " + str(f1_score(prediction_subset, label_subset)))
