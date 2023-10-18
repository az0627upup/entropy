import pandas as pd
import scipy.spatial
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ridge_regression


# Topology-based Mass Functions (TopMF)
class TopMF:

    def __init__(self, mass_type, features, labels):
        self.mass_type = mass_type
        self.features = features
        self.labels = labels
        self.mass = []

    def calculate_mass(self):
        for i in range(len(self.features)):
            if(self.mass_type == 'SEP'):
                # self.mass.append(self.separation(self.features[i],self.labels[i]))
                self.mass.append(self.separation_src(i, self.labels[i]))
            elif(self.mass_type == 'COH'):
                self.mass.append(self.cohesion(self.features[i],self.labels[i]))
            elif(self.mass_type == 'WPC'):
                self.mass.append(self.weighted_per_class(self.features[i],self.labels[i]))
            elif(self.mass_type == 'CC'):
                self.mass.append(self.circled_by_its_own_class(self.features[i],self.labels[i]))
            else:
                raise KeyError('Mass function not implemented')
        return self.mass

    def Calculate_L2(self, X, Y, c):
        # model = Lasso(alpha=0.001)
        # model = ridge_regression(alpha = 0.001)
        # model = LogisticRegression(penalty='l2',C = c)
        # W_number= np.zeros((X.shape[1], X.shape[1]))
        model = ridge_regression(X, Y, alpha=c)
        return model

    def separation_src(self, i, class_q):
        number = 0.001
        feature = np.transpose(self.features)
        w = self.Calculate_L2(feature, feature[:, i], number)
        # w[i] = 1
        sum_distance = sum([distance if class_ != class_q else 0 for class_, distance in zip(self.labels, w)])
        return sum_distance
        # mass = np.log2(sum_distance)
        # print(mass)


    def separation(self, xq, class_q):
        sumOfDistances = 0
        distances = scipy.spatial.distance.cdist(xq.reshape(1,-1), self.features, 'euclidean')
        sumOfDistances = sum([distance if class_ != class_q else 0 for class_,distance in zip(self.labels, distances[0])])
        mass = 1/(np.log2(sumOfDistances))
        return mass

    def cohesion(self, xq, class_q):
        sumOfDistances = 0
        distances = scipy.spatial.distance.cdist(xq.reshape(1,-1), self.features, 'euclidean')
        sumOfDistances = sum([distance if class_ == class_q else 0 for class_,distance in zip(self.labels, distances[0])])
        mass = 1/(np.log2(sumOfDistances))
        return mass

    def weighted_per_class(self, xq, class_q):
        n_q = sum([1 if label == class_q else 0 for label in self.labels])

        unique_elements, counts_elements = np.unique(self.labels, return_counts=True)
        class_and_freqs = sorted([(class_, freq) for class_,freq in zip(unique_elements, counts_elements)], key=lambda x: x[1])
        class_maj = class_and_freqs[-1][0]
        M = class_and_freqs[-1][1]

        mass = np.log2((M / n_q) + 1)
        return mass

    def circled_by_its_own_class(self, xq, class_q):
        distances =  scipy.spatial.distance.cdist(xq.reshape(1,-1), self.features, 'euclidean')
        class_and_distances = sorted([(class_, distance) for class_,distance in zip(self.labels, distances[0])], key=lambda x: x[1])

        SNk = sum([1 if elem[0] == class_q else 0 for elem in class_and_distances[:7]])
        
        mass = np.log2(SNk + 2)
        return mass


if __name__ == "__main__":
    df = pd.read_csv("C:/Users/az294/Desktop/02/AM-Projeto-main/datasets/iris/iris.csv",header=None)
    features, labels = df.iloc[1:, 0:-1].to_numpy(), df.iloc[1:,-1].to_numpy()
    mass = TopMF('SEP', features, labels)
    # mass.pr()
    print(mass.calculate_mass())
