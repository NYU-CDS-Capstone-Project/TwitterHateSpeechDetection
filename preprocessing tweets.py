
# coding: utf-8

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


def remove_stop_words(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def clean_tweet(tweet):
    # Remover trialing whitespace
    tweet = tweet.strip()

    # Remove @ mention
    tweet = re.sub(r'RT @[A-Za-z0-9]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'&amp+', '', tweet)  # Remove the @ mention

    # Remove Hyperlinks
    tweet = re.sub(r"http\S+", "", tweet).lower()

    # Remove Punctuation
    tweet = ''.join(c for c in tweet if c not in string.punctuation)

    # Remove Common English Words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]

    return filtered_tweet


nltk.download('stopwords')


frame = pd.read_csv("combined_tweets.csv")

tweets=frame["Tweet"].tolist()

clean_tweets= []
for tweet in tweets:
    z= clean_tweet(str(tweet))
    clean_tweets.append(z)
    

frame['clean_tweet'] = [" ".join(x) for x in clean_tweets]
frame['ID'] = range(0, frame.shape[0])

frame.to_csv('clean_data.csv')

