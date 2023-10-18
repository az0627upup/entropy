'This file is used to control the step of program.'
import Read_Data as RD
from sklearn.model_selection import RepeatedStratifiedKFold
import Entropy_Gravity as EG
import Compared_Method as CM
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import Write_CSV as WC
from tqdm import tqdm
from data_utils import *


def task(features, labels, number_of_neigh):
    single_pre = {'gravity': [], 'knn': [], 'lg': [], 'svm': [], 'nb': [], 'dt': [], 'rf': [], 'ad': [], 'nc': [], 'sgd': []}
    single_real_label = []
    simple_features = features.to_numpy()
    simple_labels = labels.to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0)
    for train_index, test_index in rskf.split(features, labels):
        X_train, y_train = simple_features[train_index], simple_labels[train_index]
        X_test, y_test = simple_features[test_index], simple_labels[test_index]
        cls = EG.Gravity(number_of_neigh)
        single_real_label.append(y_test)
        single_pre['gravity'].append(cls.classifier_by_simility(X_train, y_train, X_test))
        cm = CM.ComparedMethod()
        cm.compared(X_test, y_train, X_train)
        for key in cm.pre_dic:
            single_pre[key].append(cm.pre_dic[key])
    return single_pre, single_real_label

if __name__ == '__main__':

    data = Iris()
    data1 = Wine()
    data2 = Pima()
    data3 = Sonar()
    features, labels = data3.get_data()
    K = [0.9, 0.02, 0.32, 0.76, 0.48, 0.3, 0.46, 0.02, 0.3, 0.06, 0.5]
    i_k = 0
    all_pre_label = {}
    all_real_label = {}
    res = []
    index_dic = {'accuracy': True, 'kappa': False, 'f1_score': True, 'precision': True, 'recall': True, 'hamming_loss': False}
    all_pre_label["sonar"], all_real_label["sonar"] = task(features, labels, 0.9)
    wc = WC.CSV(all_pre_label, all_real_label, K)
    wc.write(accuracy=True, kappa=False, f1_score=True, precision=True, recall=True, hamming_loss=False)



