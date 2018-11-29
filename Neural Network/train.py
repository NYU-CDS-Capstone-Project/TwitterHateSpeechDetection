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
from torch.nn.utils import clip_grad_norm_
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

data = VectorizeData(train_sub, word2index, label = label, maxlen = seq_len)
torch.manual_seed(0)
dl = DataLoader(data, batch_size = 32, shuffle = True, num_workers = 4, drop_last = True)
val = VectorizeData(validation, word2index, label = label, maxlen = seq_len)

def get_ratio_of_classes(label):
    return ([train[label].value_counts()[1]/train[label].value_counts()[0], 1])

emb_dim = 200
hidden_size = 400
learning_rate = 0.001
batch_size = 32
weight_balance = torch.Tensor(get_ratio_of_classes(label))

net = LSTMClassifier(weights, weights.shape[0], emb_dim, hidden_size, 2, batch_size)

# Loss and Optimizer
criterion = nn.NLLLoss(weight_balance)  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

losses = []
val_scores = [0.0]

num_epochs = 5

num_batch = len(dl)
# Train the Model
for epoch in range(num_epochs):
    it = iter(dl)
    # Loop over all batches
    for i in range(num_batch):
        batch_x,batch_y,lengths = next(it)
        batch_x,batch_y,lengths = sort_batch(batch_x,batch_y,lengths)
        tweets = Variable(batch_x.transpose(0,1))
        labels = Variable(batch_y)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(tweets, lengths)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        clip_grad_norm_(net.parameters(), max_norm = 1)
        for p in net.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        optimizer.step()

        if (i+1) % 6 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, num_batch, loss.item()))
        if (i+1) % 12 == 0:
            net.eval()
            torch.manual_seed(0)
            dl2 = DataLoader(val, batch_size = 32, shuffle = False, num_workers = 0, drop_last = True)
            predictions, val_labels = get_validation_predictions(dl2, net)
            val_score = f1_score(predictions, val_labels)
            if val_score > max(val_scores):
                torch.save(net.state_dict(), 'best_rnn_' + label + '.pt')
                print ('New Val Score ' + str(val_score))
                print ('Best Net Updated Epoch ' + str(epoch + 1) + ' Iteration ' + str(i + 1))
            val_scores.append(val_score)
            net.train()

print ("best validation f-score " + str(max(val_scores)))

best_net = LSTMClassifier(weights, weights.shape[0], emb_dim, hidden_size, 2, batch_size)
best_net.load_state_dict(torch.load('best_rnn_' + label + '.pt'))
best_net.eval()

torch.manual_seed(0)
dl2 = DataLoader(val, batch_size = 32, shuffle = False, num_workers = 0, drop_last = True)

final_predictions, val_labels = get_validation_predictions(dl2, best_net)

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


