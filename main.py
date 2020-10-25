# %load main.py
import pandas as pd
import numpy as np
import _pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack


from preprocess import *



def get_stream(path, size):
    for chunk in pd.read_csv(path, chunksize=size):
        yield chunk





train_auc, val_auc = [], []
# we use one batch for training and another for validation in each iteration
# hashvec = HashingVectorizer(n_features=2**20, 
#                         preprocessor=preprocessor, tokenizer=tokenizer_stem_nostop)
# clf = SGDClassifier(loss='log', tol=1e-3)
# load from disk
hashvec = pkl.load(open('output/hashvec.pkl', 'rb'))
clf = pkl.load(open('output/clf-sgd.pkl', 'rb'))

classes = np.array([-1, 1])
epoch = 1
train_size = 27643
batch_size = 200

X_len = np.load('./dataset/len.npy')
X_img = np.load('./dataset/image.npy')
X_weekend = np.load('./dataset/weekend.npy')
for epoch_iter in range(epoch):
    print("epoch %d:" % epoch_iter)
    iters = int((train_size+batch_size-1)/(batch_size*2))
    stream = get_stream(path='./dataset/train.csv', size=batch_size)
    count = 0
    for i in range(iters):
        batch = next(stream)
        X_train, y_train = batch['Page content'], batch['Popularity']
        if X_train is None:
            break
        X_train = hashvec.transform(X_train)

        # add hand-designed feature
        batch_len = X_len[count:count+X_train.shape[0]]
        batch_img = X_img[count:count+X_train.shape[0]]
        batch_weekend = X_weekend[count:count+X_train.shape[0]]
        count += X_train.shape[0]
        X_train = hstack((X_train, batch_len, batch_img, batch_weekend), format='csr')

        clf.partial_fit(X_train, y_train, classes=classes)
        train_auc.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
        
        # validate
        batch = next(stream)
        X_val, y_val = batch['Page content'], batch['Popularity']
        X_val = hashvec.transform(X_val)

        # add hand-designed feature
        batch_len = X_len[count:count+X_val.shape[0]]
        batch_img = X_img[count:count+X_val.shape[0]]
        batch_weekend = X_weekend[count:count+X_val.shape[0]]
        count += X_val.shape[0]
        X_val = hstack((X_val, batch_len, batch_img, batch_weekend), format='csr')

        score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
        val_auc.append(score)
        print('[{}/{}] {}'.format((i+1)*(batch_size*2), train_size, score))
    batch_size = batch_size * 2
# dump to disk
pkl.dump(hashvec, open('output/hashvec.pkl', 'wb'))
pkl.dump(clf, open('output/clf-sgd.pkl', 'wb'))




