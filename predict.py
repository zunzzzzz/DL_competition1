# import csv
# import _pickle as pkl
# import pandas as pd
# import numpy as np
# num_of_model = 4
# # load from disk
# hashvec = []
# clf = []
# y_test = []
# df_test = pd.read_csv('./dataset/test.csv')
# for i in range(num_of_model):
#     hashvec_tmp = pkl.load(open('output/hashvec%d.pkl' % i, 'rb'))
#     hashvec.append(hashvec_tmp)

# for i in range(num_of_model):
#     clf_tmp = pkl.load(open('output/clf-sgd%d.pkl' % i, 'rb'))
#     clf.append(clf_tmp)


# for i in range(num_of_model):
#     y_test_tmp = clf[i].predict_proba(hashvec[i].transform(df_test['Page content']))[:,1]
#     y_test.append(y_test_tmp)




# # print(clf.predict(hashvec.transform(df_test['Page content']))[0])
# with open('./output/output.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Id', 'Popularity'])
#     for index, data in enumerate(y_test[0]):
#         accu = 0
#         for i in range(num_of_model):
#             accu += y_test[i][index]
#         average = accu / num_of_model
#         writer.writerow([df_test['Id'][index], average])


import csv
import _pickle as pkl
import pandas as pd
import numpy as np
num_of_model = 4
# load from disk
df_test = pd.read_csv('./dataset/test.csv')
hashvec_tmp = pkl.load(open('output/hashvec3.pkl' , 'rb'))

clf_tmp = pkl.load(open('output/clf-sgd3.pkl', 'rb'))

y_test_tmp = clf_tmp.predict_proba(hashvec_tmp.transform(df_test['Page content']))[:,1]





# print(clf.predict(hashvec.transform(df_test['Page content']))[0])
with open('./output/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Popularity'])
    for index, data in enumerate(y_test_tmp):
        writer.writerow([df_test['Id'][index], data])




