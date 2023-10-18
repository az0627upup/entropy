import numpy as np
import scipy
from collections import Counter
from mass.Top_mass_functions import TopMF
from mass.GP_mass_functions import GPMF
from sklearn.linear_model import Lasso
from sklearn.linear_model import ridge_regression


class SRC_strength:
    def __init__(self, k,):
        self.k = k

    def fit(self, X, Y, mass_type=None, dataset_name=None):
        """
        X:训练样本的特征, Y:训练样本的标签
        :param X:
        :param Y:
        :param mass_type:
        :param dataset_name:
        :return:
        """
        self.X = X
        self.Y = Y
        if mass_type == 'GP':
            self.mass = GPMF(dataset_name, X, Y)
        else:
            self.mass = TopMF(mass_type, X, Y)

        self.massX = self.mass.calculate_mass()

    def predict(self, XTest, result):
        finalOutput = []
        for i in range(len(XTest)):
            d = []
            votes = []
            for j in range(len(self.X)):
                strength = result[i][j]
                d.append([strength, j])
            d.sort(key=lambda tup: tup[0], reverse=True)
            d = d[0:self.k]
            for distance, j in d:
                votes.append(self.Y[j])
            answer = Counter(votes).most_common(1)[0][0]
            finalOutput.append(answer)
        return np.array(finalOutput)

    def calculate_Similarity(self, X_train, X_label, Y_test):
        number = 0.001
        result = []
        count = len(Y_test)
        X_train = np.transpose(X_train)
        Y_test = np.transpose(Y_test)
        for i in range(count):
            w = self.Calculate_L2(X_train, Y_test[:, i], number)
            w_re = list(w)
            result.append(w_re)
        return result

    def Calculate_L2(self, X, Y, c):
        # model = Lasso(alpha=0.001)
        # model = ridge_regression(alpha = 0.001)
        # model = LogisticRegression(penalty='l2',C = c)
        # W_number= np.zeros((X.shape[1], X.shape[1]))
        model = ridge_regression(X, Y, alpha=c)
        return model

    def count_distance(self, X_train, X_test, attribute_weights):
        train_length = len(X_train)
        distance = [[0] * len(X_test) for _ in range(train_length)]
        for i in range(train_length):
            x_train_single = X_train[i]
            sign = 0
            for x_test_single in X_test:
                distance[i][sign] = np.sqrt(np.sum(np.square(x_train_single - x_test_single) * attribute_weights))
                # distance[i][sign] = np.sqrt(np.sum(np.square(x_train_single - x_test_single)))
                sign += 1
        return np.array(distance)