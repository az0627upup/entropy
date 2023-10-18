from kss import KSS
from data_utils import *
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


if __name__ == '__main__':
    data = Sonar()
    data1 = Iris()
    data2 = Ecoli()
    features, labels = data2.get_data()
    simple_features = features.to_numpy()
    # simple_features = normalise_data(features)
    simple_labels = labels.to_numpy()
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(features, labels):
        train_X, train_y = simple_features[train_index], simple_labels[train_index]
        test_X, test_y = simple_features[test_index], simple_labels[test_index]
        # X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.2, random_state=4)
        kss = KSS(3)
        kss.fit(train_X, train_y, 'GP', dataset_name="sx")
        y_pred = kss.predict(test_X)
        print(metrics.classification_report(test_y, y_pred))
