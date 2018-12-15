# coding: utf-8

import pandas as pd
import re
from sklearn import model_selection
import glob
import re
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
import string
import matplotlib
import matplotlib.pyplot as plt
import nltk

#reading data

b1 = pd.read_csv("Coded Tweets (Aggregated)/Tweets Batch 1 (N=1000)_ALLRATERS.csv")
b2 = pd.read_csv("Coded Tweets (Aggregated)/Tweets Batch 2 (N=2000)_ALLRATERS.csv")
b3 = pd.read_csv("Coded Tweets (Aggregated)/Tweets Batch 3 (N=2000)_ALLRATERS.csv")
b4 = pd.read_csv("Coded Tweets (Aggregated)/Tweets Batch 4 (N=1000)_ALLRATERS.csv")
b5 = pd.read_csv("Coded Tweets (Aggregated)/Tweets Batch 5 (N=1000)_ALLRATERS.csv")

#cleaning column names

b1 = b1.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b2 = b2.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b3 = b3.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b4 = b4.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b5 = b5.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})

#concatenate batches
frame = pd.concat([b1, b2, b3, b4, b5], axis = 0)

def clean_tweet(tweet):
    # Remover trailing whitespace
    tweet = tweet.strip()
    
    tweet = re.sub("&amp;", "&", tweet)
    tweet = re.sub("&lt;", "<", tweet)
    tweet = re.sub("&gt;", ">", tweet)

    # Replace @ mention
    tweet = re.sub(r'@[A-Za-z0-9_]+', 'x_user_mention', tweet)  # Replace the @ mention

    # Replace Hyperlinks
    tweet = re.sub(r"http\S+", "x_url", tweet)
    
    # Remove hashtag
    tweet = re.sub("#", "", tweet)

    # Separate Punctuation
    tweet = word_tokenize(tweet)
    tweet = ' '.join(c for c in tweet)

    #Lowercase
    tweet = tweet.lower()
    
    #Remove trailing whitespace after transformation
    tweet = tweet.strip()

    return tweet

tweets=frame["Tweet"].tolist()

clean_tweets= []
for tweet in tweets:
    z= clean_tweet(str(tweet))
    clean_tweets.append(z)
    
frame['clean_tweet'] = clean_tweets

labels = ['CAPS', 'Obscenity', 'Threat', 'hatespeech', 'namecalling', 'negprejudice', 'noneng', 'porn', 'stereotypes']

for label in labels:
    cols = [label + str(x) for x in range(1,8)]
    frame[label + '_num_yes'] = frame[cols].sum(axis = 1)
    frame[label + '_num_non_empty'] = frame[cols].count(axis=1)
    frame[label + '_pct'] = frame[label + '_num_yes']/frame[label + '_num_non_empty']
    frame[label] = pd.Series(frame[label + '_num_yes'] >= 2).astype(int)

frame.to_csv('clean_data_2.csv')

train, test = model_selection.train_test_split(frame, test_size = 0.2, random_state = 123)

# these steps alter the number of rows in the dataframe, so they must be run AFTER the split
# to avoid small changes in the clean_tweet function affecting the split of the data
train = train[train['clean_tweet'] != ""]
train = train[train.clean_tweet.isnull() == False]
train['ID'] = range(0, train.shape[0])
train.reset_index(inplace = True, drop = True)

test = test[test['clean_tweet'] != ""]
test = test[test.clean_tweet.isnull() == False]
test['ID'] = range(0, test.shape[0])
test.reset_index(inplace = True, drop = True)

train.to_csv("train_nn_2.csv")

test.to_csv("test_nn_2.csv")

train_sub, validation = model_selection.train_test_split(train, test_size = 0.2, random_state = 123)
train_sub.reset_index(inplace = True, drop = True)
validation.reset_index(inplace = True, drop = True)

#train_sub.to_csv("train_nn_2.csv", index = False, header = False)
#validation.to_csv("val_nn_2.csv", index = False, header = False)


