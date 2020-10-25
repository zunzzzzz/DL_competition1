# %load main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocess import *


df = pd.read_csv('./dataset/train.csv')

############################ length of article ##########################

# len_array = np.zeros((df.shape[0], 1))
# for i in range(df.shape[0]):
#     text = df.iloc[i, 2]
#     text = preprocessor(text)
#     x = tokenizer_stem_nostop(text)
#     # print(len(x))
#     len_array[i] = len(x)
# # len_array = np.load('./dataset/len.npy')
# print(len_array[:10])
# np.save('./dataset/len.npy', len_array)


# len_array = np.load('./dataset/len.npy')
# sc_x = StandardScaler()
# len_array = sc_x.fit_transform(len_array)
# print(len_array[:10])
# np.save('./dataset/len.npy', len_array)


########################### weekend ######################################
# weekend_array = np.zeros((df.shape[0], 1))
# for i in range(df.shape[0]):
# # for i in range(1):
#     text = df.iloc[i, 2]
#     text = text.lower()
#     if ('datetime="sat' in text) or ('datetime="sun' in text):
#         label = df.iloc[i, 1]
#         weekend_array[i][0] = 1

# # weekend_array = np.load('./dataset/weekend.npy')
# np.save('./dataset/weekend.npy', weekend_array)

######################## image href ############################################
# img_array = np.zeros((df.shape[0], 1))

# for i in range(df.shape[0]):
# # for i in range(1):
#     text = df.iloc[i, 2]
#     # text = df.iloc[5000, 2]
#     text = text.lower()
#     # print(text.count('<img') + text.count('href='), df.iloc[i, 1])
#     img_array[i][0] = text.count('<img') + text.count('href=')

# sc_x = StandardScaler()
# img_array = sc_x.fit_transform(img_array)
# print(img_array[:50])
# np.save('./dataset/image.npy', img_array)



######################## test ############################################
weekend_array = np.zeros((df.shape[0], 1))
# for i in range(df.shape[0]):
count_total = 0
count_true = 0
string = 'ebola '
# print(df.shape[0])
# for i in range(df.shape[0]):
for i in range(1):
    text = df.iloc[i, 2]
    text = df.iloc[24050, 2]
    text = text.lower()
    print(text)
    if string in text:
        label = df.iloc[i, 1]
        # print(i, label)
        count_total += 1
        if label == 1:
            count_true += 1

print(string)
print(count_true)
print(count_total)
print(count_true / count_total)
