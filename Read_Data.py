import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ReadData(object):
    '''
    This class is used to get the information of dataset, such as the matrix of data, number of data and attributes.
    The format of original datasets should be

    1,1,1,1,0
    2,1,3,1,1
    .......

    an object is represented by a row of data;
    the last attribute (last col) should be the label and others are conditional attributes;
    attributes should be patition by a comma;
    '''
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.__data_matrix = None
        self.__label_matrix = None
        self.__number_of_data = 0
        self.__number_of_attributes = 0
        self.read_original_datasets()
        # self.__class_number = collections.Counter(self.__label_matrix)

    def read_original_datasets(self):
        std=MinMaxScaler()
        data = np.loadtxt(self.dataset_name, delimiter=',')
        self.__number_of_attributes = len(data[0])-1
        self.__number_of_data = len(data)
        self.__data_matrix = std.fit_transform(np.delete(data, -1, axis=1))
        self.__label_matrix = np.transpose(np.array(data[:, -1]))


    @ property
    def data_matrix(self):
        return self.__data_matrix

    @property
    def label_matrix(self):
        return self.__label_matrix

    @property
    def number_of_data(self):
        return self.__number_of_data

    @property
    def number_of_attributes(self):
        return self.__number_of_attributes

    @property
    def class_number(self):
        return self.__class_number