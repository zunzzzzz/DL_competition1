# %load main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')

# train
len_array = np.zeros((df.shape[0], 1))
for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    len_array[i] = len(text)

sc_x = StandardScaler()
sc_x.fit(len_array)
len_array = sc_x.transform(len_array)
print(len_array[:10])
np.save('./dataset/len.npy', len_array)

# test
len_array = np.zeros((df_test.shape[0], 1))
for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    len_array[i] = len(text)
len_array = sc_x.transform(len_array)
np.save('./dataset/len_test.npy', len_array)
print(len_array[:10])