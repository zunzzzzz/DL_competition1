# %load main.py
import pandas as pd
import numpy as np
import _pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from preprocess import *

# def get_stream(path, size):
#     for chunk in pd.read_csv(path, chunksize=size):
#         yield chunk

df = pd.read_csv('./dataset/train.csv')
df_feature = pd.read_csv('./dataset/feature.csv')


X_feature = df_feature.to_numpy()

# X, X_val, y, y_val = train_test_split(
#     df['Page content'], df['Popularity'], test_size=0.1, random_state=0)

X = df['Page content'][:25000]
X_val = df['Page content'][25000:]

y = df['Popularity'][:25000]
y_val = df['Popularity'][25000:]

classes = np.array([-1, 1])
epoch = 100
train_size = X.shape[0]
batch_size = 5000


train_auc, val_auc = [], []
# we use one batch for training and another for validation in each iteration
hashvec = HashingVectorizer(n_features=2**20, 
                       preprocessor=preprocessor, tokenizer=tokenizer_stem_nostop)
clf = SGDClassifier(loss='log', tol=1e-3, alpha=1e-7)
# # load from disk
# hashvec = pkl.load(open('output/hashvec%d.pkl' % batch_size, 'rb'))
# clf = pkl.load(open('output/clf-sgd%d.pkl' % batch_size, 'rb'))



# for epoch_iter in range(epoch):
#     print("epoch %d:" % epoch_iter)
#     iters = int((train_size+batch_size-1)/(batch_size))
#     # print(iters)
#     count = 0
#     for i in range(iters):
#         if i == iters - 1:
#             X_train, y_train = X[count:], y[count:]
#         else:
#             X_train, y_train = X[count:count+batch_size], y[count:count+batch_size]

#         X_train = hashvec.transform(X_train)
#         # hstack((X_train, X_feature[count:count+X_train.shape[0]]))

#         count += X_train.shape[0]

#         clf.partial_fit(X_train, y_train, classes=classes)
#         train_score = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
#         train_auc.append(train_score)
        
#         # validate
#         X_val_tmp = hashvec.transform(X_val)
#         # hstack((X_val_tmp, X_feature[25000:]))
#         val_score = roc_auc_score(y_val, clf.predict_proba(X_val_tmp)[:,1])
#         val_auc.append(val_score)
#         print('[{}/{}] train_score = {} val_score = {}'.format(count, train_size, train_score, val_score))
#     # dump to disk
#     pkl.dump(hashvec, open('output/hashvec%d.pkl' % batch_size, 'wb'))
#     pkl.dump(clf, open('output/clf-sgd%d.pkl' % batch_size, 'wb'))

# tfidf
tfidf = TfidfVectorizer(preprocessor=preprocessor, 
                    tokenizer=tokenizer_stem_nostop)
# clf = SGDClassifier(loss='log', tol=1e-3, alpha=1e-7)
clf = SVC(probability=True, kernel='rbf')
# clf = RandomForestClassifier(max_depth=5, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=100)
# pipe = Pipeline([('vect', TfidfVectorizer(preprocessor=preprocessor, 
#                                             tokenizer=tokenizer_stem_nostop)), 
#                     ('clf', clf)])

X = tfidf.fit_transform(X)
print(X.shape)
# X = hstack((X, X_feature[:25000]))
print(X.shape)
X_val = tfidf.transform(X_val)
# X_val = hstack((X_val, X_feature[25000:]))
clf.fit(X, y)

train_score = roc_auc_score(y, clf.predict_proba(X)[:,1])
train_auc.append(train_score)

val_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
val_auc.append(val_score)
print('train_score = {} val_score = {}'.format(train_score, val_score))
# dump to disk
pkl.dump(clf, open('output/clf.pkl', 'wb'))
pkl.dump(tfidf, open('output/tfidf.pkl', 'wb'))

