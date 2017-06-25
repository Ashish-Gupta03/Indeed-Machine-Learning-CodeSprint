import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from data_preprocessing import join_strings
from sklearn.metrics import confusion_matrix
from model import  count_vectorizer_test_x, tfidf_vectorizer_test_x, file_cnt, file_tfidf

count_vectorizer_model, tfidf_vectorizer_model = joblib.load(file_cnt), joblib.load(file_tfidf)
print("Both the trained models have been imported successfully!")
print()
print("Making predictions...")
pred1 = count_vectorizer_model.predict(count_vectorizer_test_x.toarray())
pred2 = tfidf_vectorizer_model.predict(tfidf_vectorizer_test_x.toarray())

# Combine predictions and map the labels if the values do not equal 0, else assign empty string
# arr = np.where((pred1 + pred2) != 0, mlb.classes_, "")
print ('pred1 ',pred1)
print ('pred2 ',pred2)
# Load the array into a DataFrame constructor and join non-empty strings
# predictions = pd.DataFrame(arr).apply(join_strings, axis=1).to_frame("tags")
# predictions['tags2'] = predictions['tags'].str.split(' ').str[0]

# predictions['tags1'] = pred1
# predictions['tags2'] = pred2
# print ('predictions ',predictions.head())
# DataFrame.get_value(index, 'tags', takeable=False)
# Submit predictions
print("Submitting predictions...")
s = pd.Series(pred2)
s.to_csv("tags.tsv", index=False)
newLi2 = []
newLi3 = []
for i in pred2:
	if i == '':
		i = 0
	newLi2.append(int(i))

df4 = pd.read_csv('testSyne.csv')
for i in df4['tags']:
	newLi3.append(int(i))	
print ('newli3 ',newLi3)	
print ('accuracy is ',accuracy_score(newLi2,newLi3))
print ('confusion matrix ',confusion_matrix(newLi2,newLi3))
print("done")