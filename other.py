# %load main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')



text = df.iloc[650]['Page content']
text = preprocessor(text)
text = tokenizer_stem_nostop(text)
print(text)

# count = 0
# count_t = 0
# for i in range(df.shape[0]):
#     text = df.iloc[i]['Page content']
#     text = text.lower()

#     if '/category/entertain' in text:
#         count += 1 
#         if df.iloc[i]['Popularity'] == 1:
#             count_t += 1
# print(count)
# print(count_t / count)
# category_array = sc_x.fit_transform(category_array)
# print(category_array[:10])
# np.save('./dataset/category_test.npy', category_array)


