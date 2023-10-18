import csv
import threading
import Metric as M
import datetime as dt
import numpy as np


class CSV(object):
    """
    This class is used to import results to csv file.
    Predict labels and real labels are needed for init.

    Example:
        csv = CSV(result)
        index_dic = {'accuracy'}:True, 'kappa':True, 'confusion_matrix': True, 'f1_score': True,
        'precision': True, 'recall': True, 'hamming_loss': True}
        csv.write(index_dic)
    """

    def __init__(self, pre_label, real_label, K):
        """
        :param result: dic, {dataset1:{method1:[[prelabel],[reallabel],.....], method2:....}, dataset2:{...}}
        """
        self.pre_label_dic = pre_label
        self.real_label_dic = real_label
        self.dataset_name = list(pre_label.keys())
        self.method_name = list(self.pre_label_dic[self.dataset_name[0]].keys())
        self.k = K
        self.judge = M.Metric()

    def write(self, **kwargs):
        """
        This function is a controller uses multi-thread to write csv of different index.
        :param kwargs: index dic
        :return: no return, all results are saved in folder: '/results' with name format: 'index+time'.
        """
        if kwargs['accuracy']:
            self.summary_to_excel('accuracy', self.judge.accuracy)
        if kwargs['kappa']:
            self.summary_to_excel('kappa', self.judge.kappa)
        if kwargs['f1_score']:
            self.summary_to_excel('f1_score', self.judge.f1_score)
        if kwargs['precision']:
            self.summary_to_excel('precision', self.judge.precision)
        if kwargs['recall']:
            self.summary_to_excel('recall', self.judge.recall)
        if kwargs['hamming_loss']:
            self.summary_to_excel('hamming_loss', self.judge.hamming_loss)

    def summary_to_excel(self, index_name, fun):
        """
        This function is used to get results of a specific index and save it to csv file.
        :param index_name: name of index
        :param fun: function of metric
        :return: no return, all results are saved in folder: '/results' with name format: 'index+time'.
        """
        result = [self.method_name[:]]
        result[0].insert(0, 'datasets')
        for row in self.dataset_name:
            temp = [row]
            for col in self.method_name:
                if self.pre_label_dic[row][col][0]!=[]:
                    temp.append(fun(np.array(self.pre_label_dic[row][col]), self.real_label_dic[row]))
            result.append(temp)
        now_time = dt.datetime.now().strftime('%F-%T')
        name = index_name+now_time
        name = name.replace(':', '-') + '--k=' + str(self.k)
        with open('E:/pythonProject/Entropy-knn/results/%s.csv' % name, 'w+', encoding='utf-8') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            for r in result:
                csv_writer.writerow(r)






