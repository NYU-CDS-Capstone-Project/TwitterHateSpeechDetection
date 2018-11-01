# Progress Memo
## 10/31/18
   
   We have several finished models and have implemented many of the strategies that seemed likely to succeed based on our
review of the literature. We aren’t yet satisfied with the performance (f-score) of the models we’ve produced. The set of
labeled tweets is on the small side, and some of the labels are much harder to predict than others.

   We have many labels to predict: obscenity, threat, hate speech, name-calling, negative prejudice, non-English, and
stereotypes. Vivienne Badaan and Mark Hoffarth from the Social Justice Lab managed the process of having the datasets
labeled. They mentioned that, in particular, labelling the tweets as to whether they contained “stereotypes” was difficult
for the raters, and the raters often didn’t agree, which is reflected in the Intraclass Correlation (ICC) metrics. We’re using
a majority vote standard for the classification problem; a tweet belongs to a category if at least half of the raters who
reviewed it labeled it as such. Our baseline model is a linear SVM, which performed reasonably well in detecting obscenity,
non-English, and hate speech, but performed poorly on many other categories. It’s possible that another architecture would
succeed on the remaining labels, but it’s also possible that the labelling is a bit inconsistent for some of the categories,
which makes the prediction problem noisier. The graph below represent the mean cross-validation F score for classifier by
category.

![alt text](https://github.com/NYU-CDS-Capstone-Project/TwitterHateSpeechDetection/blob/master/baseline_models.png)

   We’ve also built a recurrent neural network that uses word embeddings pre-trained on Twitter data. The paper [Deep
Learning for Hate Speech Detection in Tweets](https://arxiv.org/abs/1706.00188) by Badjatiya et al achieved an f-score of
0.808 with an LSTM and GLoVe embeddings. However, our implementation achieves an f-score of 0.612 detecting hate speech, lower
than the baseline model’s performance. It’s possible that the model is too complex for our small data size (3k tweets, after
reserving some tweets for test and validation). We’ve also experimented with using additional datasets. We used a [publicly
available dataset](https://data.world/crowdflower/hate-speech-identification) of 16k tweets labeled as hate speech to pre
train a model before training it on the 3k tweets from our own data. However, as currently implemented, this actually worsens
the performance.
  
   We’ve discussed the data size issue with the researchers at the Social Justice Lab, and they’ve labeled an additional
1k tweets (which we haven’t yet incorporated into our model) and promised to add 1k more in the coming weeks. This should
improve model performance. We may also try using the [Universal Language Model Fine-tuning for Text Classification]
(https://arxiv.org/abs/1801.06146), a transfer learning method which, in some cases, can produce strong results with only a
few hundred labeled examples.
