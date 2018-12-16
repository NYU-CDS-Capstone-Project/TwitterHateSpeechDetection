import pandas as pd
from sklearn import model_selection

train = pd.read_csv("../train_nn.csv")

train_sub, validation = model_selection.train_test_split(train, test_size = 0.2, random_state = 456)
train_sub.reset_index(inplace = True, drop = True)
validation.reset_index(inplace = True, drop = True)

labels = ['Obscenity', 'Threat', 'hatespeech', 'namecalling', 'negprejudice', 'noneng', 'porn', 'stereotypes']

for label in labels:
	train_sub_label = train_sub[[label, 'ID', 'clean_tweet']]
	train_sub_label.columns = ['Quality', '#1 ID', '#1 String']
	train_sub_label.to_csv("./DATA/" + label + "/train.tsv", index = False, sep = '\t')

	validation_label = validation[[label, 'ID', 'clean_tweet']]
	validation_label.columns = ['Quality', '#1 ID', '#1 String']
	validation_label.to_csv("./DATA/" + label + "/dev.tsv", index = False, sep = '\t')
