import numpy as np
import scipy
from collections import Counter
from mass.Top_mass_functions import TopMF
from mass.GP_mass_functions import GPMF


class KSS_Strength:
    def __init__(self, k,):
        self.k = k

    def fit(self, X, Y, mass_type=None, dataset_name=None):
        self.X = X
        self.Y = Y
        if mass_type == 'GP':
            self.mass = GPMF(dataset_name, X, Y)
        else:
            self.mass = TopMF(mass_type, X, Y)
        self.massX = self.mass.calculate_mass()

    def predict(self, XTest):
        finalOutput = []
        distances = scipy.spatial.distance.cdist(self.X, XTest, 'euclidean')
        # distances = self.count_distance(self.X, XTest, self.attribute_weights)
        for i in range(len(XTest)):
            d = []
            votes = []
            for j in range(len(self.X)):
                strength = (self.massX[j]) / ((np.power(distances[j][i], 2)) + 0.000000000000001)
                d.append([strength, j])
            d.sort(key=lambda tup: tup[0], reverse=True)
            d = d[0:self.k]
            for distance, j in d:
                votes.append(self.Y[j])
            answer = Counter(votes).most_common(1)[0][0]
            finalOutput.append(answer)

        return np.array(finalOutput)

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