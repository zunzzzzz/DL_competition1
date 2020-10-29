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
list_array = np.zeros((df.shape[0], 1))

for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()
    list_array[i][0] = text.count('<li')

sc_x = StandardScaler()
sc_x.fit(list_array)
list_array = sc_x.transform(list_array)
print(list_array[:10])
np.save('./dataset/list.npy', list_array)

# test
list_array = np.zeros((df_test.shape[0], 1))

for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()
    list_array[i][0] = text.count('<li')

list_array = sc_x.transform(list_array)
print(list_array[:10])
np.save('./dataset/list_test.npy', list_array)