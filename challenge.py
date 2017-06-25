import numpy as np
import pandas as pd
import csv
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

def p(x):
	# print (x)
	dek = [''.join(word) for word in x.split(' ')]
	print (dek)


df = pd.read_csv('train.tsv',sep='\t')
df2 = pd.read_csv('test.tsv',sep='\t')

df['tags'][df['tags'].isnull()==True] = ' '
df['description'] = df['description'].apply(p)
print (df.head())


# y_train = []
# for i in range(len(df['tags'])):
# 	# y_train.append(df['tags'][i])
# 	x=[]
# 	for j in df['tags'][i].split(' '):
# 		x.append(j)
# 	y_train.append(x)


# mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform(y_train)

classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3),max_df=20000,min_df=1,smooth_idf=True,norm="l2",sublinear_tf=True,use_idf=False,stop_words='english')),
    ('clf', LinearSVC(penalty="l1", dual=False, tol=1e-4))])

# print ('type x ',type(df['description']))
# print ('type y ',type(Y))
classifier.fit(df['description'], df['tags'])
predicted = classifier.predict(df2['description'])
# all_labels = mlb.inverse_transform(predicted)
# print (all_labels)
# for labels in all_labels:
#     print(' '.join(labels))
print ('accuracy score ',accuracy_score(predicted,df2['tags']))