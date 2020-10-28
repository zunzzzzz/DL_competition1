import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


####################### image  ############################################
# train 
img_array = np.zeros((df.shape[0], 1))

for i in range(df.shape[0]):
    text = df.iloc[i]['Page content']
    text = text.lower()
    img_array[i][0] = text.count('<img')

sc_x = StandardScaler()
sc_x.fit(img_array)
img_array = sc_x.transform(img_array)
print(img_array[:50])
np.save('./dataset/image.npy', img_array)

# test
img_array = np.zeros((df_test.shape[0], 1))

for i in range(df_test.shape[0]):
    text = df_test.iloc[i]['Page content']
    text = text.lower()
    img_array[i][0] = text.count('<img')

img_array = sc_x.transform(img_array)
print(img_array[:50])
np.save('./dataset/image_test.npy', img_array)