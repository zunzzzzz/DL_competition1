# %load main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import *
import re

df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


for i in range(100):
    text = df.iloc[100+i]['Page content']
    text = preprocessor(text)
    text = tokenizer_stem_nostop(text)
    print(100+i, df.iloc[100+i]['Popularity'])



