import pandas as pd
from sklearn import model_selection
import re
from nltk.tokenize import word_tokenize

terms0_3 = pd.read_csv("../Full Dataset/Terms 0-3 Final LIWC WITH Tweet Text RANDOMIZED.csv")
terms4 = pd.read_csv("../Full Dataset/Terms 4 Final LIWC WITH Tweet Text RANDOMIZED.csv")
terms5 = pd.read_csv("../Full Dataset/Terms 5-7 Final LIWC WITH tweet text RANDOMIZED.csv")

frame = pd.concat([terms0_3, terms4, terms5], axis = 0)

print ('frame shape:', frame.shape)
print ('number ids:', len(set(frame.tweet_id)))
print ('tweet order:', len(set(frame.tweet_order)))

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

tweets=frame["tweet"].tolist()

clean_tweets= []
for tweet in tweets:
    z= clean_tweet(str(tweet))
    clean_tweets.append(z)

frame['clean_tweet'] = clean_tweets

frame.loc[:, 'label'] = 0

frame = frame[['label', 'tweet_order', 'clean_tweet']]

frame.columns = ['Quality', '#1 ID', '#1 String']
frame.to_csv("DATA/test.tsv", index = False, sep = '\t')
