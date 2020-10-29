# %load main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')

digit_array = np.zeros((df.shape[0], 1))
for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()

    # title
    title_start = text.find('<h1 class="title">')
    title_end = text.find('</h1>')
    title = text[title_start+18:title_end]
    if re.match('[0-9]+', title) is not None:
        digit_array[i][0] = 1
         
print(digit_array[:10])
np.save('./dataset/digit.npy', digit_array)

digit_array = np.zeros((df_test.shape[0], 1))
for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()

    # title
    title_start = text.find('<h1 class="title">')
    title_end = text.find('</h1>')
    title = text[title_start+18:title_end]
    if re.match('[0-9]+', title) is not None:
        digit_array[i][0] = 1

print(digit_array[:10])
np.save('./dataset/digit_test.npy', digit_array)