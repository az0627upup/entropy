import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn
def normalise_data(X):
    # 数据归一化
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    # 数据标准化
    # std = StandardScaler()
    # data = std.fit_transform(X)
    return X_scaled




data = pd.read_csv("E:\pythonProject\Entropy-knn\datasets\sonar\sonar.csv", header=0)
data = np.array(data)
data = np.delete(data, 0, axis=1)
label = data[:, -1]
label_num = len(set(label.tolist()))
label_set = set(label.tolist())
features = normalise_data(np.delete(data, -1, axis=1))
# print(label)
# print(features)
samples = []
for it in zip(features, label):
    samples.append((it[0], it[1]))
# print(samples)


def distacne(a,b):
    # 计算a和b的欧式距离
    dist = np.linalg.norm(a - b)
    return dist
ave_num = np.average(features,axis=0)
dictance = -1
for it in features:
    tmp = distacne(it,ave_num)
    if tmp >= dictance:
        dictance = tmp
feature_info = []
features_entropy = []
for feature in features:
    tmp = []
    for sample in samples:
        if distacne(feature,sample[0])<=dictance :
            tmp.append(sample)
    feature_info.append(tmp[:])
for feature in feature_info:
    #初始化dict
    count = {}
    length = len(feature)
    for it in label_set:
        count[it] = 0
    for sub in feature:
        count[sub[1]] += 1
    tmp_entropy = 0
    for key,value in count.items():
        p = value/length
        try:
            tmp_entropy = tmp_entropy - p*math.log(p,math.e)
        except:
            pass
    features_entropy.append(tmp_entropy)
print(features_entropy)


# for lab in feature_info[0]:
#     print(lab[1])
#print(len(feature_info[1]))




# print(features.shape)
# print(features)
# print(np.average(features,axis=0))

# ave_col = []
# cols = list(data.columns)
# cols.pop(0) # 去id——name
# label_name = cols.pop() # 取标签

# for col in cols:
#     ave_col.append(sum(data[col])/len(data))

