import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn import grid_search

from data_preprocessing import tokenize

# Read data
train = pd.read_csv("trainSyne.csv")
test = pd.read_csv("testSyne.csv")

# Remove missing values in train
X_train = train[train['tags'].notnull()]

train_x = X_train['description'].as_matrix()  # train-description
test_x = test['description'].as_matrix()  # test-description
train_labels = X_train['tags'].as_matrix()  # train-tags

unhappy = X_train[X_train['tags'] == 0]
happy = X_train[X_train['tags'] == 1]
neutral = X_train[X_train['tags'] == 2]
print ('unhappy ',len(unhappy))
print ('happy ',len(happy))
print ('neutral ',len(neutral))

hap = len(happy) / float(len(neutral))
unhap = len(happy) / float(len(unhappy))
# For balancing both the loans, sample the data and throw away data points from  larger sample having more data points

unhappy = unhappy.sample(frac=unhap,random_state=1)
neutral = neutral.sample(frac=hap,random_state=1)
df = unhappy.append(happy).append(neutral)
train_x = df['description']
train_labels = df['tags']
# Transform train-tags into a multi-label binary format
# mlb = MultiLabelBinarizer()
# train_labels = mlb.fit_transform(train_y)

# Applying CountVectorizer on character n-grams (specifically for tri-grams range)
count_vectorizer = CountVectorizer(stop_words="english", tokenizer=tokenize, ngram_range=(1, 3),
                                   max_features=10000,token_pattern=r'\b\w+\b',max_df=20000,min_df=1)

# Learn and transform train-description
count_vectorizer_train_x = count_vectorizer.fit_transform(train_x)
count_vectorizer_test_x = count_vectorizer.transform(test_x)

# Applying TfIdfVectorizer on individual words(specifically for tri-grams range)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", tokenizer=tokenize, ngram_range=(1, 3),
                                   max_features=30000, analyzer="word",max_df=20000,min_df=1,
                                   smooth_idf=True,norm="l2",sublinear_tf=True,use_idf=False)

# Learn and transform train-description
tfidf_vectorizer_train_x = tfidf_vectorizer.fit_transform(train_x)
tfidf_vectorizer_test_x = tfidf_vectorizer.transform(test_x)

# GradientBoostingClassifier with parameter tuning
params = {"n_estimators": 170, "max_depth": 5, "random_state": 10, "min_samples_split": 4, "min_samples_leaf": 2}
# classifier = OneVsRestClassifier(GradientBoostingClassifier(**params))
# classifier = xgb.XGBClassifier(learning_rate =0.1,n_estimators=1000,\
#  max_depth=10,\
#  min_child_weight=1,\
#  gamma=0.1,\
#  subsample=0.8,\
#  colsample_bytree=0.8,\
#  nthread=4,\
#  scale_pos_weight=1,\
#  seed=27)
# classifier = xgb.XGBClassifier()
# param_grid = {
#         'scale_pos_weight' : [2, 2.5, 3, 3.5],
#         'n_estimators': [100, 150, 200],
#         'eta':[0.01,0.015,0.025,0.05,0.1],
#         'gamma':[0.1,0.3,0.5,0.9],
#         'max_depth':[3,5,7,9],
#         'max_child_weight':[1,3,5,7],
#         'subsample':[0.6,0.7,0.8,0.9],
#         'colsample_bytree':[0.6,0.7,0.8,0.9],
#         'lambda':[0.01,0.1,1.0],
#         'alpha':[0.0,0.1,0.5,1.0]
#     }

    
# classifier = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=3, verbose=20, scoring = 'f1_micro')
param_grid = {
        'scale_pos_weight' : [2, 2.5, 3, 3.5],
        'n_estimators': [100, 150, 200],
    }

classifier = GradientBoostingClassifier(max_depth=35,n_estimators=100)
# classifier = RandomForestClassifier(max_depth=20,min_samples_leaf=2)
# Generate predictions using counts
classifier.fit(count_vectorizer_train_x, train_labels)
file_cnt = "loaded_model/count_vectorizer_model.pkl"  # serialize model with pickle
os.makedirs(os.path.dirname(file_cnt), exist_ok=True)
with open(file_cnt, "w") as f:
    joblib.dump(classifier, file_cnt)
print("CountVectorizer based trained classifier ready to be exported")

# Calling fit() more than once will overwrite what was learned by any previous fit()
# Generate predictions using tf-idf representation

classifier.fit(tfidf_vectorizer_train_x, train_labels)
file_tfidf = "loaded_model/tfidf_vectorizer_model.pkl"  # serialize model with pickle
os.makedirs(os.path.dirname(file_tfidf), exist_ok=True)
with open(file_tfidf, "w") as f:
    joblib.dump(classifier, file_tfidf)
print("TfidfVectorizer based trained classifier ready to be exported")