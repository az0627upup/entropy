import Read_Data
import Feature_Weight
import numpy as np
import collections
from kss import KSS
from SRC import SRC_strength
from kss_strength import KSS_Strength
class Gravity(object):
    def __init__(self, k):
        self.attribute_weights = None
        self.predict_label = []
        self.num_of_neighbor = k
    def classifier(self, train_X, train_y, test_X):
        """
        train_X:训练样本特征, train_y:训练样本标签, test_X:测试样本特征
        :param train_X:
        :param train_y:
        :param test_X:
        :return:
        """
        self.attribute_weights = Feature_Weight.FeatureWeight(train_X, train_y, self.num_of_neighbor)()
        k_range = 6
        kss = KSS_Strength(k_range)
        kss.fit(train_X,train_y,'SEP') #初始代码
        y_pred = kss.predict(test_X)
        return y_pred

    def classifier_by_simility(self, train_X, train_y, test_X):
        """
        模糊相似度去做分类
        :param train_X:
        :param train_y:
        :param test_X:
        :return:
        """
        k_range = 6
        kss = KSS(k_range)
        kss.fit(train_X, train_y, 'SEP')
        y_pred = kss.predict_class(test_X)
        return y_pred

    def classifier_by_SRC(self, train_X, train_y, test_X):
        """
        通过SRC去做分类
        :return:
        """
        k_range = 6
        kss = SRC_strength(k_range)
        result = kss.calculate_Similarity(train_X, train_y, test_X)
        kss.fit(train_X, train_y, 'SEP')
        y_pred = kss.predict(test_X, result)
        return y_pred
