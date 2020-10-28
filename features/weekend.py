import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


# train
weekend_array = np.zeros((df.shape[0], 1))

for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()
    if ('datetime="fri' in text) or ('datetime="sat' in text) or ('datetime="sun' in text):
        weekend_array[i][0] = 1



print(weekend_array[:10])
np.save('./dataset/weekend.npy', weekend_array)


# test
weekend_array = np.zeros((df_test.shape[0], 1))

for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()
    if ('datetime="fri' in text) or ('datetime="sat' in text) or ('datetime="sun' in text):
        weekend_array[i][0] = 1

print(weekend_array[:10])
np.save('./dataset/weekend_test.npy', weekend_array)