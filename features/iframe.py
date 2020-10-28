import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


# train
iframe_array = np.zeros((df.shape[0], 1))

for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()
    iframe_array[i][0] = text.count('<iframe')


sc_x = StandardScaler()
sc_x.fit(iframe_array)
iframe_array = sc_x.transform(iframe_array)
print(iframe_array[:10])
np.save('./dataset/iframe.npy', iframe_array)


# test
iframe_array = np.zeros((df_test.shape[0], 1))

for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()
    iframe_array[i][0] = text.count('<iframe')


iframe_array = sc_x.transform(iframe_array)
print(iframe_array[:10])
np.save('./dataset/iframe_test.npy', iframe_array)