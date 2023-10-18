import Read_Data
from sklearn.neighbors import NearestNeighbors


class FeatureWeight(object):
    '''
    This class is uesd to calculate the weight of every conditional attribute.

    :param
    train_X: arr-like; the matrix of training samples
    train_y: arr-like; the matrix of labels
    number_of_neigh: the number of neighbors

    :output
    attributes_weight: arr-like; the weights of attributes

    :example
    train_X = [[1,2,3][2,3,4]]
    train_y = [1,2]
    number_of_neigh = 3
    fw = FeatureWeight(train_X, train_y, number_of_neigh)
    print(fw())
    output-----> [0.1, 0.9]
    '''

    def __init__(self, train_X, train_y, number_of_neigh):
        self.label_matrix = train_y
        self.data_matrix = train_X
        self.number_of_attributes = len(train_X[0])
        self.__attributes_weight = []
        self.__k = number_of_neigh

    def cal_neigh_of_data(self, data_under_single_attribute):   #计算每个样本的每个属性的k个邻居
        '''

        This function is used to calculate the neighbors of every data under an attibute.
        The module of NearestNeighbors in sklearn is used in there, more information can be find in
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
        :param:data_under_single_attribute: arr-like, one col, multi row, a specific attribute values of data
        :return: array-like: neighbors' index of every data under an attibute.
        '''
        if isinstance(self.__k, int):
            nbrs = NearestNeighbors(n_neighbors=self.__k + 1).fit(data_under_single_attribute)
            return nbrs.kneighbors(data_under_single_attribute, return_distance=False)
        else:
            nbrs = NearestNeighbors(radius=self.__k).fit(data_under_single_attribute)
            return nbrs.radius_neighbors(data_under_single_attribute, return_distance=False)

    def cal_emd_of_single_attribute(self, data_under_single_attribute):   #计算每个样本的每个属性的邻居
        '''
        This function is uesd to calcute the sum of all neighbor under a conditional attribute.
        :param data_under_single_attribute: arr-like, only have one col, multi row, a specific attribute values of data
        :return: real-value, sum of EMD
        '''
        sum_of_EMD = 0
        neighbor_index = self.cal_neigh_of_data(data_under_single_attribute)
        for element in neighbor_index:
            patition_of_neigh = self.partition(element)
            if patition_of_neigh != -1:
                sum_of_EMD += self.cal_emd_of_single_neigh(set(element), patition_of_neigh)
        return sum_of_EMD

    def cal_emd_of_single_neigh(self, set1, set2):
        '''
        This function is used to calculate the emd between a set and its partition.
        :param set1: arr-like, such as [1,2,3]
        :param set2: arr-like, partition of set1, such as [[1,2],[3]]
        :return: real-value, emd between set1 and set2.
        '''
        emd_of_neigh = 0
        for equ_class in set2:
            unionset = set1.union(set(equ_class))
            intersectset = set1.intersection(set(equ_class))
            emd_of_neigh += (len(intersectset) * (len((unionset - intersectset)) / len(unionset)))
        return emd_of_neigh

    def partition(self, neighbor_index):
        '''
        This function is used to product a partition of a neighbor.
        :param neighbor_index: arr-like, the index of neighbor
        :return: a partition of a neighbor
        '''
        label_dic = {}
        for index in neighbor_index:
            if self.label_matrix[index] not in label_dic:
                label_dic[self.label_matrix[index]] = [index]
            else:
                label_dic[self.label_matrix[index]].append(index)
        if len(label_dic) == 1:
            return -1
        else:
            return list(label_dic.values())

    def normalize_emd_to_weight(self, emd_of_attributes):
        '''
        This function is used to normalize the emd of attributes to the weights.
        :param emd_of_attributes: arr-like, emd of every attributes.
        :return: arr-like, normalized weights.
        '''
        weights = []
        max_val = max(emd_of_attributes)
        for value in emd_of_attributes:
            weights.append(1-(value/max_val)+0.1)
        return weights

    def normal_att_weight(self):
        a = max(self.__attributes_weight)
        arr1 = []
        for _ in range(len(self.__attributes_weight)):
            arr1.append((1 - (self.__attributes_weight[_] / a))+0.1)
        return arr1

    def __call__(self, *args, **kwargs):
        '''
        This fuction is the top function of this class.
        :param args:
        :param kwargs:
        :return: weights of attributes.
        '''
        for i in range(self.number_of_attributes):
            self.__attributes_weight.append(self.cal_emd_of_single_attribute(self.data_matrix[:, [i]]))
        return self.normal_att_weight()
