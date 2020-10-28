import numpy as np
import csv

# train
X_len = np.load('./dataset/len.npy')
X_img = np.load('./dataset/image.npy')
X_href = np.load('./dataset/href.npy')
X_iframe = np.load('./dataset/iframe.npy')
X_weekend = np.load('./dataset/weekend.npy')

X = np.concatenate((X_len, X_img, X_href, X_iframe, X_weekend), axis=1)
X = X.tolist()
print(len(X))
with open('./dataset/feature.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['length of article', 'number of images', 'number of hrefs', 'number of iframes', 'is weekend'])
    writer.writerows(X)

# test
X_len = np.load('./dataset/len_test.npy')
X_img = np.load('./dataset/image_test.npy')
X_href = np.load('./dataset/href_test.npy')
X_iframe = np.load('./dataset/iframe_test.npy')
X_weekend = np.load('./dataset/weekend_test.npy')

X = np.concatenate((X_len, X_img, X_href, X_iframe, X_weekend), axis=1)
X = X.tolist()
print(len(X))
with open('./dataset/feature_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['length of article', 'number of images', 'number of hrefs', 'number of iframes', 'is weekend'])
    writer.writerows(X)