# %load main.py
import pandas as pd
import numpy as np
import _pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from preprocess import *
import os, itertools, csv



df = pd.read_csv('./dataset/train.csv')
df_feature = pd.read_csv('./dataset/feature.csv')




X, y = df['Page content'], df['Popularity']
X_feature = df_feature

# count vector 
tfidf = TfidfVectorizer(
                preprocessor=preprocessor, 
                tokenizer=tokenizer_stem_nostop)


X = tfidf.fit_transform(X)
X = hstack((X, X_feature))
print(X.shape)


clf = LogisticRegression(solver = "liblinear")
print('start fitting')
clf.fit(X, y)
print('finish fitting')


scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring='roc_auc')
print('%.3f (+/- %.3f)' % (scores.mean(), scores.std()))
# print('with features: train_score = {} val_score = {}'.format(train_score, val_score))

# dump to disk
# pkl.dump(clf, open('output/clf-content-feature-svc-test.pkl', 'wb'))
# pkl.dump(tfidf, open('output/count-content-feature-test.pkl', 'wb'))

