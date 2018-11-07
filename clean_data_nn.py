# coding: utf-8

import pandas as pd
import re
from sklearn import model_selection
import glob
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
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

#nltk.download('stopwords')

def remove_stop_words(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def clean_tweet(tweet):
    # Remover trailing whitespace
    tweet = tweet.strip()

    # Remove @ mention
    tweet = re.sub(r'RT @[A-Za-z0-9:_]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'&amp+', '', tweet)  # Remove the @ mention

    # Remove Hyperlinks
    tweet = re.sub(r"http\S+", "", tweet).lower()
    
    # Remove hashtag
    tweet = re.sub("#", " ", tweet)

    # Separate Punctuation
    tweet = nltk.tokenize.word_tokenize(tweet)
    tweet = ' '.join(c for c in tweet)
    
    #Remove trailing whitespace after transformation
    tweet = tweet.strip()

    return tweet

tweets=frame["Tweet"].tolist()

clean_tweets= []
for tweet in tweets:
    z= clean_tweet(str(tweet))
    clean_tweets.append(z)
    
frame['clean_tweet'] = clean_tweets
frame['ID'] = range(0, frame.shape[0])

frame.reset_index(inplace = True, drop = True)

frame.to_csv('clean_data_nn.csv')

train, test = model_selection.train_test_split(frame, test_size = 0.2, random_state = 123)

train = train[train['Tweet'].isnull() == False]

train.to_csv("train_nn.csv")

test.to_csv("test_nn.csv")

