from kss import KSS
from data_utils import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from test_simple import Tests
data = Wine()
# data1 = Glass()
data2 = Vehicle()
data4 = Sonar()
data5 = Wine()
features, labels = data4.get_data()
X_train,X_test,y_train,y_test = train_test_split(features.to_numpy(),labels.to_numpy(),test_size=0.2,random_state=4)



# print(X_train)
# print(X_train[0])
# test_simple_distance = Tests(X_train,X_test)
# m = test_simple_distance.countdistance()
# n = test_simple_distance.countdistances()
# print(m)
# print(n)


k_range = range(1, 7)
for k in k_range:
  kss = KSS(k)
  kss.fit(X_train,y_train,'GP',dataset_name="sx")
  # kss.fit(X_train,y_train,'CC')
  y_pred = kss.predict(X_test)
  print(k)
  print(metrics.classification_report(y_test, y_pred))