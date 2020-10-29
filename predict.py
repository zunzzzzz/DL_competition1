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



clf = pkl.load(open('output/tfidf_svc.pkl', 'rb'))
tfidf = pkl.load(open('output/tfidf.pkl', 'rb'))

X_test = df_test['Page content']
X_test = tfidf.transform(X_test)
X_test = hstack((X_test, X_feature))

y_test = (clf.predict_proba(X_test)[:,1])

with open('./output/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Popularity'])
    for index, data in enumerate(y_test):
        writer.writerow([df_test['Id'][index], data])




