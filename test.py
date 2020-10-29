# %load main.py
import pandas as pd
import numpy as np
import _pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('./dataset/train.csv')

sc_x = StandardScaler()

df_feature = pd.read_csv('./dataset/feature.csv')

X_feature = df_feature.to_numpy()
X = X_feature
print(X_feature.shape)


X_train, X_val, y_train, y_val = train_test_split(
    X, df['Popularity'], test_size=0.03, random_state=0)





clf = SVC(probability=True, random_state=0, kernel='rbf')
# clf = KNeighborsClassifier(n_neighbors=100)
# clf = SGDClassifier(loss='log', max_iter=10000)
# clf = DecisionTreeClassifier(random_state=0)
# clf = RandomForestClassifier(max_depth=5, random_state=0)


clf.fit(X_train, y_train)
# clf = pkl.load(open('output/clf-svc.pkl', 'rb'))
train_score = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
val_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
print('train_score = {}, val_score = {}'.format(train_score, val_score))
pkl.dump(clf, open('output/svc.pkl', 'wb'))