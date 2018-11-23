#!/usr/bin/python

"""to use, write python create_weights.py filename matrix_id where filename is the name
of the data file (must contain a column called clean_tweet) and matrix_id is part of the filename
for the matrix"""

import torch
import pandas as pd
import numpy as np
from sklearn import model_selection
from utils import *
from model import *
from data_loader import *
import pickle
import sys

#maybe make a parameter where you can specify a weights matrix file instead of having to build one?
#should splitting the datasets occur elsewhere? like in the clean data nn file?

print ('filename:', sys.argv[1])
print ('matrix id:', sys.argv[2])

filename = sys.argv[1]
matrix_id = sys.argv[2]

train = pd.read_csv("../" + str(filename))

train = train[train.clean_tweet.isnull() == False]

train_sub, validation = model_selection.train_test_split(train, test_size = 0.2, random_state = 123)
train_sub.reset_index(inplace = True, drop = True)
validation.reset_index(inplace = True, drop = True)

vocab_size = 10000

vocab = build_vocab(vocab_size, train_sub.clean_tweet)
word2index, index2word = build_idx(vocab)

glove_path = "/Users/carolineroper/Desktop/Capstone Project/Neural Network/glove.twitter.27B/glove.twitter.27B.200d.txt"

glove = load_glove(glove_path)

my_glove_path = "/Users/carolineroper/Desktop/Capstone Project/glove/vectors.txt"

custom_glove = load_glove(my_glove_path)

weights_matrix = build_weights_matrix(glove, custom_glove, index2word, 200)
weights = torch.FloatTensor(weights_matrix)

matrix_filename = 'weights_matrix_' + str(matrix_id) + '.sav'
pickle.dump(weights, open(matrix_filename, "wb"))
