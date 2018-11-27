
# coding: utf-8

import pandas as pd
from sklearn import model_selection
import re
from nltk.tokenize import word_tokenize

terms_0_3 = pd.read_csv("../Full Dataset/Terms 0-3 Final LIWC WITH Tweet Text RANDOMIZED.csv")

terms_5_7 = pd.read_csv("../Full Dataset/Terms 5-7 Final LIWC WITH tweet text RANDOMIZED.csv")

terms_4 = pd.read_csv("../Full Dataset/Terms 4 Final LIWC WITH tweet text RANDOMIZED.csv")

all_tweets = pd.concat([terms_0_3, terms_4, terms_5_7], axis = 0)
all_tweets = all_tweets.loc[pd.isnull(all_tweets.tweet)==False]

def clean_tweet(tweet):
    # Remover trailing whitespace
    tweet = tweet.strip()

    # Remove @ mention
    tweet = re.sub(r'RT @[A-Za-z0-9:_]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove the @ mention
    
    #Fix html punctuation
    tweet = re.sub("&amp;", "&", tweet)
    tweet = re.sub("&lt;", "<", tweet)
    tweet = re.sub("&gt;", ">", tweet)

    # Remove Hyperlinks
    tweet = re.sub(r"http\S+", "", tweet).lower()
    
    # Remove hashtag
    tweet = re.sub("#", " ", tweet)

    # Separate Punctuation
    tweet = word_tokenize(tweet)
    tweet = ' '.join(c for c in tweet)
    
    #Remove trailing whitespace after transformation
    tweet = tweet.strip()

    return tweet

all_tweets['clean_tweet'] = [clean_tweet(x) for x in all_tweets.tweet]
all_tweets[['clean_tweet']].to_csv("../glove/all_tweets_cleaned.txt", index = False, header = False)

