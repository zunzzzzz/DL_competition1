import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer

from preprocess import tokenizer_stem_nostop

df = pd.read_csv('./dataset/train.csv')
print(df.head(5))

pipe = Pipeline([('vect', HashingVectorizer(n_features=2**10,
                                             preprocessor=preprocessor, 
                                             tokenizer=tokenizer_stem_nostop)), 
                  ('clf', LogisticRegression(solver = "liblinear"))])
        
scores = cross_val_score(estimator=pipe, X=df_small['review'], y=df_small['sentiment'], \
                         cv=10, scoring='roc_auc')
print('LogisticRegression+preprocess+hash: %.3f (+/-%.3f)' % (name, scores.mean(), scores.std()))