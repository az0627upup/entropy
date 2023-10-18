from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
import sklearn.metrics as skm
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from Read_Data import ReadData

class ComparedMethod(object):
    '''
    This function is used to implement the classifier compared to Gravity. Most method is imported from sklearn,
    more information, see https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    '''

    def __init__(self):
        self.__pre_dic = {'knn': [],
                          'lg': [],
                          'svm': [],
                          'nb': [],
                          'dt': [],
                          'rf': [],
                          'ad': [],
                          'nc':[],
                          'sgd':[]}

    def knn(self, test_X, train_y, train_X, k):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_X, train_y)
        self.__pre_dic['knn'] = clf.predict(test_X)

    def logist(self, test_X, train_y, train_X):
        clf = LogisticRegression(solver='liblinear', multi_class='auto')
        clf.fit(train_X, train_y)
        self.__pre_dic['lg'] = clf.predict(test_X)

    def svm(self, test_X, train_y, train_X, c=1):
        clf = SVC(C=c, kernel='rbf', probability=True, gamma='auto')
        clf.fit(train_X, train_y)
        self.__pre_dic['svm'] = clf.predict(test_X)

    def naive_bayes(self, test_X, train_y, train_X):
        clf = GaussianNB()
        clf.fit(train_X, train_y)
        self.__pre_dic['nb'] = clf.predict(test_X)

    def decision_tree(self, test_X, train_y, train_X):
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_y)
        self.__pre_dic['dt'] = clf.predict(test_X)

    def random_forest(self, test_X, train_y, train_X):
        clf = RandomForestClassifier(criterion='entropy', random_state=0)
        clf.fit(train_X, train_y)
        self.__pre_dic['rf'] = clf.predict(test_X)

    def adj48(self, test_X, train_y, train_X):
        clf = AdaBoostClassifier()
        clf.fit(train_X, train_y)
        self.__pre_dic['ad'] = clf.predict(test_X)

    def nearest_center(self, test_X, train_y, train_X):
        clf = NearestCentroid()
        clf.fit(train_X, train_y)
        self.__pre_dic['nc'] = clf.predict(test_X)


    def SGD(self, test_X, train_y, train_X):
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        clf.fit(train_X, train_y)
        self.__pre_dic['sgd'] = clf.predict(test_X)

    def compared(self, test_X, train_y, train_X, c=1, k=3):
        self.knn(test_X, train_y, train_X, k)
        self.logist(test_X, train_y, train_X)
        self.svm(test_X, train_y, train_X, c)
        self.naive_bayes(test_X, train_y, train_X)
        self.decision_tree(test_X, train_y, train_X)
        self.random_forest(test_X, train_y, train_X)
        self.adj48(test_X, train_y, train_X)
        self.nearest_center(test_X, train_y, train_X)
        self.SGD(test_X, train_y, train_X)

    @property
    def pre_dic(self):
        return self.__pre_dic



if __name__ == '__main__':
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    # rd = ReadData('./datasets/bupa.txt')
    rd = ReadData('./datasets/pima.txt')
    pre = []
    real = []
    k = 0
    for train_index, test_index in kf.split(rd.data_matrix, rd.label_matrix):
        train_X, train_y = rd.data_matrix[train_index], rd.label_matrix[train_index]
        test_X, test_y = rd.data_matrix[test_index], rd.label_matrix[test_index]
        cm = ComparedMethod()
        cm.knn(test_X, train_y, train_X, 3)
        k += 1
        print(k)
        print(cm.pre_dic)

