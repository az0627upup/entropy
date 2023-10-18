import sklearn.metrics as sm

class Metric(object):
    """
    This class is used to evaluate the results of classifiers.
    All the index implemented here are from the module of sklearn. More information about it, see:
    https://scikit-learn.org/stable/modules/classes.html?highlight=metric#module-sklearn.metrics

    :example
    m = Metric()
    result = m.accuracy([[1,2,3,4],[2,2,2,2]], [[1,2,1,1],[2,2,2,2]])

    """


    def accuracy(self, pre_label, real_label):
        results = []
        for i in range(len(real_label)):
            results.append(sm.accuracy_score(real_label[i], pre_label[i]))
        return sum(results)/len(results)

    def kappa(self, pre_label, real_label):
        results = [sm.cohen_kappa_score(real_label[i], pre_label[i]) for i in range(len(real_label))]
        return sum(results)/len(results)

    # def confusion_matrix(self, pre_label, real_label):
    #     results = [sm.confusion_matrix(real_label[i], pre_label[i]) for i in range(len(real_label))]
    #     return results

    def f1_score(self, pre_label, real_label):
        results = [sm.f1_score(real_label[i], pre_label[i], average='macro') for i in range(len(real_label))]
        return sum(results) / len(results)

    def precision(self, pre_label, real_label):
        results = [sm.precision_score(real_label[i], pre_label[i], average='macro') for i in
                   range(len(real_label))]
        return sum(results) / len(results)

    def recall(self, pre_label, real_label):
        results = [sm.recall_score(real_label[i], pre_label[i], average='macro') for i in
                   range(len(real_label))]
        return sum(results) / len(results)

    def hamming_loss(self, pre_label, real_label):
        results = [sm.hamming_loss(real_label[i], pre_label[i]) for i in range(len(real_label))]
        return sum(results) / len(results)

