import pandas as pd
from sklearn import model_selection

train = pd.read_csv("../train_nn.csv")
test = pd.read_csv("../test_nn.csv")

labels = ['Obscenity', 'Threat', 'hatespeech', 'namecalling', 'negprejudice', 'noneng', 'porn', 'stereotypes']

for label in labels:
	train_label = train[[label, 'ID', 'clean_tweet']]
	train_label.columns = ['Quality', '#1 ID', '#1 String']
	train_label.to_csv("./BERT/DATA/" + label + "/train.tsv", index = False, sep = '\t')

	test_label = test[[label, 'ID', 'clean_tweet']]
	test_label.columns = ['Quality', '#1 ID', '#1 String']
	test_label.to_csv("./BERT/DATA/" + label + "/dev.tsv", index = False, sep = '\t')
