from pandas import DataFrame
import pandas as pd
import nltk, re, pprint
from nltk import word_tokenize
from collections import defaultdict

df = pd.read_csv('train.tsv',sep='\t')
df.head(n=5)

df2 = pd.read_csv("test.tsv", sep="\t")
df2.head(n=2)

tags = ["part-time-job", "full-time-job", "hourly-wage", "salary", "associate-needed",
"bs-degree-needed", "ms-or-phd-needed", "licence-needed", "1-year-experience-needed",
"2-4-years-experience-needed", "5-plus-years-experience-needed", "supervising-job"]

def preproc(tag, df, df2):

    x_train = []
    y_train = []
    for index, row in df.iterrows():
    	x_train.append(row.description.lower())

    	if type(row.name) == str and tag in row.name:
            y_train.append(1)
    	else:
            y_train.append(0)


        # if type(row.name) == str and  tag in row.name:
        #     y_train.append(1)
        # else:
        #     y_train.append(0)
        
    docs_new = []
    for index, row in df2.iterrows():
    	docs_new.append(row.description.lower())
            
    return x_train, y_train, docs_new

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
    
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
# from mlxtend.classifier import StackingClassifier
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# https://github.com/scikit-learn/scikit-learn/issues/1156
# Snowball stemmers could be used as a dependency
from nltk.stem import SnowballStemmer

class build_stemmer(object):
    def __init__(self):
        self.wns = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.wns.stem(t) for t in word_tokenize(doc)]    
    
# http://www.nltk.org/howto/stem.html 
# https://github.com/nltk/nltk/issues/1581
# pip install nltk==3.2.1
from nltk.stem.porter import *

class build_stemmer2(object):
    def __init__(self):
        self.wns = PorterStemmer()
    def __call__(self, doc):
        return [self.wns.stem(t) for t in word_tokenize(doc)]        
    
class lemma_stemmer(object):
    def __init__(self):
        self.wns = SnowballStemmer('english')
        self.wnl = WordNetLemmatizer() 
    def __call__(self, doc):
        return [self.wnl.lemmatize(self.wns.stem(t)) for t in word_tokenize(doc)]    
    
    
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import grid_search
import random
random.seed(2017)

def pred(tag, x_train, y_train, docs_new):
    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    #count_vect = CountVectorizer()
    count_vect = CountVectorizer(tokenizer=build_stemmer(), ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(x_train)
    X_train_counts.shape
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape


    clf = xgb.XGBClassifier()
    '''
    param_grid = {
        'n_estimators': [500],
        'learning_rate': [0.05],
        'max_depth': [5, 7],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.75, 0.85],
    }
    '''

    param_grid = {
        'scale_pos_weight' : [2, 2.5, 3, 3.5],
        'n_estimators': [100, 150, 200],
    }

    
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=3, verbose=20, scoring = 'f1_micro')

    model.fit(X_train_tfidf, y_train)    

    
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    

    predicted = model.predict(X_new_tfidf)


    print(sum(predicted))
    return predicted

outputs = [""] * 2921

for i, tag in enumerate(tags):
    print ('init tag ',tag)
    x_train, y_train, docs_new  = preproc(tag, df, df2)

    output = pred(tag, x_train, y_train, docs_new)
    
    for j, item in enumerate(output):
        if item == 1:
            if outputs[j] == "":
                outputs[j] = tag
            else:
                outputs[j] += " " + tag
            
for i, item in enumerate(outputs):
    if item == "":
        outputs[i] = " "

import csv

with open("tags_xgboost_weight-cv.tsv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["tags"]) 
    for val in outputs:
        wr.writerow([val])

print("Done")