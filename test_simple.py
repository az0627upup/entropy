import scipy
import numpy as np
class Tests:
    def __init__(self, Xtrain, Xtest):
        self.X = Xtrain
        self.Xtest = Xtest

    def countdistance(self):
        distance = scipy.spatial.distance.cdist(self.X, self.Xtest, 'euclidean')
        return distance
    def countdistances(self):
        train_length = len(self.X)
        distance = [[0] * len(self.Xtest) for _ in range(train_length)]
        for i in range(train_length):
            x_train_single = self.X[i]
            sign = 0
            for x_test_single in self.Xtest:
                distance[i][sign] = np.sqrt(np.sum(np.square(x_train_single - x_test_single)))
                # distance[i][sign] = np.square(x_train_single, x_test_single) * attribute_weights
                # distance[i][sign] = np.sum(np.square(x_train_single, x_test_single))
                sign += 1
        return np.array(distance)


