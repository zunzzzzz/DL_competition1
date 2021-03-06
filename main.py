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




# X, X_val, y, y_val = train_test_split(
#     df['Page content'], df['Popularity'], test_size=0.3, random_state=0)
# X_feature, X_feature_val, _, _ = train_test_split(
#     df_feature, df['Popularity'], test_size=0.3, random_state=0)

X, y = df['Page content'], df['Popularity']
X_feature = df_feature
# tfidf
# tfidf = TfidfVectorizer(preprocessor=preprocessor, 
#                     tokenizer=tokenizer_stem_nostop)
# count vector 
tfidf = TfidfVectorizer(preprocessor=preprocessor, 
                    tokenizer=tokenizer_stem_nostop)

# clf = SVC(probability=True, kernel='rbf')


X = tfidf.fit_transform(X)
X = hstack((X, X_feature))
print(X.shape)
# X_val = tfidf.transform(X_val)
# X_val = hstack((X_val, X_feature_val))

clf = SVC(probability=True, kernel='rbf')
print('start fitting')
clf.fit(X, y)
print('finish fitting')
train_score = roc_auc_score(y, clf.predict_proba(X)[:,1])
# val_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])


# print('train_score = {} val_score = {}'.format(train_score, val_score))
print('train_score = {}'.format(train_score))
# dump to disk
pkl.dump(clf, open('output/tfidf_svc.pkl', 'wb'))
pkl.dump(tfidf, open('output/tfidf.pkl', 'wb'))

