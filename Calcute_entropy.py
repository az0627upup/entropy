import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn
from data_utils import *


def distacne(a,b):
    # 计算a和b的欧式距离
    dist = np.linalg.norm(a - b)
    return dist

def normalise_data(X):
    # 数据归一化
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    # 数据标准化
    # std = StandardScaler()
    # data = std.fit_transform(X)
    return X_scaled


def calcute_entropy(features, label):
    label_num = len(set(label.tolist()))
    label_set = set(label.tolist())
    samples = []
    for it in zip(features, label):
        samples.append((it[0], it[1]))
    ave_num = np.average(features, axis=0)
    dictance = -1
    for it in features:
        tmp = distacne(it, ave_num)
        if tmp >= dictance:
            dictance = tmp
    feature_info = []
    features_entropy = []
    for feature in features:
        tmp = []
        for sample in samples:
            if distacne(feature, sample[0]) <= dictance:
                tmp.append(sample)
        feature_info.append(tmp[:])
    for feature in feature_info:
        # 初始化dict
        count = {}
        length = len(feature)
        for it in label_set:
            count[it] = 0
        for sub in feature:
            count[sub[1]] += 1
        tmp_entropy = 0
        for key, value in count.items():
            p = value / length
            try:
                tmp_entropy = tmp_entropy - p * math.log(p, math.e)
            except:
                pass
        features_entropy.append(tmp_entropy)
    return features_entropy
    # print(features_entropy)
    # print(len(features_entropy))


if __name__ == '__main__':
    data = Wine()
    features, labels = data.get_data()
    simple_features = features.to_numpy()
    # simple_features = normalise_data(features)
    simple_labels = labels.to_numpy()
    m = calcute_entropy(simple_features, simple_labels)
    print(m)
    print(len(m))
