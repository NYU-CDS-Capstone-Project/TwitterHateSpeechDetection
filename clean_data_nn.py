
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

#cleaning column names

b1 = b1.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b2 = b2.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})
b3 = b3.rename(columns = {'hatespeech7namecalaling1':'hatespeech7', 'nanmecalling2':'namecalling2'})

#cleaning labels

r1 = pd.read_csv("Coded Tweets (Aggregated)/rater_key_1.csv")
r2 = pd.read_csv("Coded Tweets (Aggregated)/rater_key_2.csv")
r3 = pd.read_csv("Coded Tweets (Aggregated)/rater_key_3.csv")

r1['Rater'] = r1['Rater'].str.strip()
r2.columns = r1.columns
r3.columns = r1.columns

r = r1.merge(r2, how = 'left', on = "Rater")
r = r.merge(r3, how = 'left', on = "Rater")
r .columns = ['Number_1', 'Rater', "Number_2", 'Number_3']
r.loc[r['Rater'] == "Parker", 'Number_3'] = 7

#dictionary of correspondences for rater tables

dict2_name = r[["Number_2", "Rater"]].set_index("Number_2")['Rater'].to_dict()
dict3_name = r[["Number_3", "Rater"]].set_index("Number_3")['Rater'].to_dict()

#convert rater numbers to rater names

for i in range(1,8):
    b2.columns = [re.sub(str(i), dict2_name[i], x) for x in b2.columns.values]

for i in range(1,8):
    b3.columns = [re.sub(str(i), dict3_name[i], x) for x in b3.columns.values]

#convert rater names to rater numbers

dict_name_1 = r[["Rater", "Number_1"]].set_index("Rater")["Number_1"]

for i in r['Rater'].tolist():
    b2.columns = [re.sub(i, str(dict_name_1[i]), x) for x in b2.columns.values]

for i in r['Rater'].tolist():
    b3.columns = [re.sub(i, str(dict_name_1[i]), x) for x in b3.columns.values]

#concatenate batches
frame = pd.concat([b1, b2, b3], axis = 0)

#nltk.download('stopwords')

def remove_stop_words(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def clean_tweet(tweet):
    # Remover trialing whitespace
    tweet = tweet.strip()

    # Remove @ mention
    tweet = re.sub(r'RT @[A-Za-z0-9:]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove the @ mention
    tweet = re.sub(r'&amp+', '', tweet)  # Remove the @ mention

    # Remove Hyperlinks
    tweet = re.sub(r"http\S+", "", tweet).lower()

    # Remove Punctuation
    tweet = nltk.tokenize.word_tokenize(tweet)
    tweet = ' '.join(c for c in tweet)

    # Remove Common English Words (commenting out because not wanted for nn)
    #stop_words = set(stopwords.words('english'))
    #word_tokens = word_tokenize(tweet)
    #filtered_tweet = [w for w in word_tokens if not w in stop_words]

    return tweet

tweets=frame["Tweet"].tolist()

clean_tweets= []
for tweet in tweets:
    z= clean_tweet(str(tweet))
    clean_tweets.append(z)
    
frame['clean_tweet'] = clean_tweets
frame['ID'] = range(0, frame.shape[0])

frame.to_csv('clean_data_nn.csv')

train, test = model_selection.train_test_split(frame, test_size = 0.2, random_state = 123)

train = train[train['Tweet'].isnull() == False]

train.to_csv("train_nn.csv")

test.to_csv("test_nn.csv")

