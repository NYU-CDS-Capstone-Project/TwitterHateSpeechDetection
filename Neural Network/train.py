# coding: utf-8

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

#drew inspiration from
#https://github.com/dmesquita/understanding_pytorch_nn and
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
#https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch.ipynb
#https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
#https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb

# ## <b> Data Processing<b>

label = 'hatespeech'

train = pd.read_csv("../train_nn.csv")

train_sub, validation = model_selection.train_test_split(train, test_size = 0.2, random_state = 123)
train_sub.reset_index(inplace = True, drop = True)
validation.reset_index(inplace = True, drop = True)

vocab_size = 10000

vocab = build_vocab(vocab_size, train_sub.clean_tweet)
word2index, index2word = build_idx(vocab)

glove_path = "/Users/carolineroper/Desktop/Capstone Project/Neural Network/glove.twitter.27B/glove.twitter.27B.200d.txt"

glove = load_glove(glove_path)

weights_matrix = build_weights_matrix(glove, index2word, 200)
weights = torch.FloatTensor(weights_matrix)

data = VectorizeData(train_sub, word2index, label = label)
dl = DataLoader(data, batch_size = 32, shuffle = True)
val = VectorizeData(validation, word2index, label = label)
dl2 = DataLoader(val, batch_size = 32, shuffle = False)

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths

def get_validation_predictions(validation_data_loader, model):
    predictions = []
    pred_labels = []
    #get training predictions
    it = iter(validation_data_loader)
    num_batch = len(validation_data_loader) - 1
    # Loop over all batches
    for i in range(num_batch):
        batch_x,batch_y,batch_len = next(it)
        batch_x,batch_y,batch_len = sort_batch(batch_x,batch_y,batch_len)
        tweets = Variable(batch_x.transpose(0,1))
        batch_labels = Variable(batch_y)
        lengths = batch_len.numpy()
        outputs = model(tweets, lengths)
        _, pred = torch.max(outputs.data, 1)
        predictions.extend(list(pred.numpy()))
        pred_labels.extend(list(batch_labels.data.numpy()))
    return predictions, pred_labels

def get_ratio_of_classes(label):
    return ([train[label].value_counts()[1]/train[label].value_counts()[0], 1])

hidden_size = 200
learning_rate = 0.001
batch_size = 32
weight_balance = torch.Tensor(get_ratio_of_classes(label))

net = LSTMClassifier(weights, weights_matrix.shape[0], hidden_size, hidden_size, 2, batch_size)

# Loss and Optimizer
criterion = nn.NLLLoss(weight_balance)  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

losses = []
val_scores = []

num_epochs = 5

num_batch = len(dl) - 1
# Train the Model
for epoch in range(num_epochs):
    it = iter(dl)
    # Loop over all batches
    for i in range(num_batch):
        batch_x,batch_y,batch_len = next(it)
        batch_x,batch_y,batch_len = sort_batch(batch_x,batch_y,batch_len)
        tweets = Variable(batch_x.transpose(0,1))
        labels = Variable(batch_y)
        lengths = batch_len.numpy()
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(tweets, lengths)
        loss = criterion(outputs, labels)
        losses.append(loss.data[0])
        loss.backward()
        clip_grad_norm(net.parameters(), max_norm = 1)
        for p in net.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        optimizer.step()

        if (i+1) % 6 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train.clean_tweet)//batch_size, loss.data[0]))
        if (i+1) % 12 == 0:
            predictions, val_labels = get_validation_predictions(dl2, net)
            val_scores.append(f1_score(predictions, val_labels))
            if val_scores[-1] == max(val_scores):
                best_net = net
                print ('Best Net Updated Epoch ' + str(epoch + 1) + ' Iteration ' + str(i + 1))

print ("best validation f-score" + str(max(val_scores)))

filename = 'best_rnn_' + label + '.sav'
pickle.dump(best_net, open(filename, "wb"))

final_predictions, val_labels = get_validation_predictions(dl2, best_net)

print ("confusion matrix" + str(confusion_matrix(val_labels, final_predictions)))

'''if __name__ == "  __main__":
    parser = argparse.ArgumentParser()

    # Data loading parameters
    parser.add_argument('--label', type=str, default='hatespeech')
    config = parser.parse_args()
    print (config)
    label = config[0]'''

