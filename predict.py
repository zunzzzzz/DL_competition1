import csv
import _pickle as pkl
import pandas as pd
import numpy as np
from preprocess import *
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


df_test = pd.read_csv('./dataset/test.csv')
df_feature = pd.read_csv('./dataset/feature_test.csv')

X_feature = df_feature.to_numpy()


# load hand-designed features

# X_test = np.concatenate((X_len, X_title, X_img, X_weekend, X_href, X_iframe, X_time), axis=1)

# hashvec = pkl.load(open('output/hashvec1000.pkl', 'rb'))
clf = pkl.load(open('output/adv_clf.pkl', 'rb'))
tfidf = pkl.load(open('output/adv_tfidf.pkl', 'rb'))
# pipe = pkl.load(open('output/tfidf.pkl', 'rb'))
clf_feature = pkl.load(open('output/svc.pkl', 'rb'))
# X_test= np.concatenate((X_len, X_title, X_img, X_weekend), axis=1)
# y_test = clf.predict_proba(X_test)[:,1]
X_test = df_test['Page content']
X_test = tfidf.transform(X_test)

X_test_feature = X_feature

# y_test = ((clf.predict_proba(X_test)[:,1]) + (clf_feature.predict_proba(X_test_feature)[:,1])) / 2
y_test = (clf.predict_proba(X_test)[:,1])

# print(clf.predict(hashvec.transform(df_test['Page content']))[0])
with open('./output/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Popularity'])
    for index, data in enumerate(y_test):
        writer.writerow([df_test['Id'][index], data])




