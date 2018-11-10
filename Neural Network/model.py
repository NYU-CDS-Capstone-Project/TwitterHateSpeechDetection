#!/usr/bin/env python

"""The model itself: an LSTM with a single layer and dropout"""

import torch.nn
import torch.autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(torch.nn.Module):
    def __init__(self, weights, vocab_size, embedding_dim, hidden_dim, output_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.weights = weights
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(self.weights)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.hidden2out = torch.nn.Linear(hidden_dim, output_size)
        self.softmax = torch.nn.LogSoftmax()
        self.dropout_layer = torch.nn.Dropout(p=0.2)
        self.batch_size = batch_size
    
    def init_hidden(self, batch_size):
        return(torch.autograd.Variable(torch.randn(1, self.batch_size, self.hidden_dim)),                
            torch.autograd.Variable(torch.randn(1, self.batch_size, self.hidden_dim)))

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.embedding(batch) 
        embeds = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)
        outputs, lengths = pad_packed_sequence(outputs)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)
        return output
