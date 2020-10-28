import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


######################### href ########################################
# train 
href_array = np.zeros((df.shape[0], 1))

for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()
    href_array[i][0] = text.count('href=')

sc_x = StandardScaler()
sc_x.fit(href_array)
href_array = sc_x.transform(href_array)
print(href_array[:50])
np.save('./dataset/href.npy', href_array)

# test
href_array = np.zeros((df_test.shape[0], 1))

for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()
    href_array[i][0] = text.count('href=')

href_array = sc_x.transform(href_array)
print(href_array[:50])
np.save('./dataset/href_test.npy', href_array)