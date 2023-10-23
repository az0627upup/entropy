import numpy as np
import scipy
import multiprocessing as mp
import Feature_Weight
from collections import Counter
from mass.Top_mass_functions import TopMF
from mass.GP_mass_functions import GPMF
from Calcute_entropy import *
from sklearn.cluster import KMeans
from sklearn import preprocessing

class KSS:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y, mass_type=None, dataset_name=None):
        """
        :param X:  训练样本的特征
        :param Y:  训练样本的标签
        :param mass_type:
        :param dataset_name:
        :return:
        """
        self.X = X
        self.Y = Y
        if mass_type == 'GP':
            self.mass = GPMF(dataset_name, X, Y)
        else:
            self.mass = TopMF(mass_type, X, Y)
        self.massX = self.mass.calculate_mass()
        self.massX = self.mass.calculate_mass()
        # self.massX = 0.01 * np.array(self.mass.calculate_mass()) + 0.01 * np.array(calcute_entropy(self.X, self.Y))

    def predict(self, XTest):
        finalOutput = []
        # distances = scipy.spatial.distance.cdist(self.X, XTest, 'euclidean')
        attribute_weights = Feature_Weight.FeatureWeight(self.X, self.Y, 3)()
        print(attribute_weights)
        distances = self.count_distance(self.X, XTest, attribute_weights)
        # x = np.concatenate((self.X, XTest))
        # attr = self.compute_similarity_before(x)   # 计算样本的相似度
        # temp = np.sqrt(np.sum(np.square(self.X - XTest) * self.attribute_weights))
        for i in range(len(XTest)):         # 测试样本
            d = []
            votes = []
            for j in range(len(self.X)):    # 训练样本
                strength = (self.massX[j])/((np.power(distances[j][i], 2))+0.000000000000001)
                # strength_attr = (self.massX[j]) * attr
                d.append([strength, j])
            d.sort(key=lambda tup: tup[0], reverse=True)
            d = d[0:self.k]
            for distance, j in d:
                votes.append(self.Y[j])
            answer = Counter(votes).most_common(1)[0][0]
            finalOutput.append(answer)
        return np.array(finalOutput)

    def predict_neighbor(self, XTest):
        """
        求测试样本与训练样本中每类前k个近邻，最后求测试样本关于这K个近邻的引力和，最大的引力就属于哪一类
        :param XTest:
        :return:
        """
        finalOutput = []
        # distances = scipy.spatial.distance.cdist(self.X, XTest, 'euclidean')
        attribute_weights = Feature_Weight.FeatureWeight(self.X, self.Y, 3)()
        print(attribute_weights)
        distances = self.count_distance(self.X, XTest, attribute_weights)
        for i in range(len(XTest)):
            d = []
            vas = []
            for j in range(len(self.X)):
                vas.append((self.Y[j], distances[j][i]))
                print(vas)
            # 使用列表来存储各个类别的元素和对应的下标
            color_elements = {}
            # 遍历数据，将元素和下标存储在列表中
            for index, (color, value) in enumerate(vas):
                if color not in color_elements:
                    color_elements[color] = []
                color_elements[color].append((value, index))
            # 找出各类别排名前四的元素的下标
            top_two_indices = {}
            for color, elements in color_elements.items():
                elements.sort(reverse=True)
                top_two_indices[color] = [element[1] for element in elements[-4:]]
            # 输出各类别排名最后的四个元素的下标
            for color, indices in top_two_indices.items():
                print(f"类别: {color}, 排名前四的元素的下标: {indices}")
            for key, values in top_two_indices.items():
                strength = 0
                for value in values:
                    strength += ((self.massX[value])/((np.power(distances[value][i], 2))+0.000000000000001))
                d.append([strength, key])
            print(d)
            max_value = max(d, key=lambda x: x[0])
            max_category = max_value[1]
            finalOutput.append(max_category)
            print(finalOutput)

    def calculate_similarity(self, fea_train, fea_test):
        """
        计算相似度
        :param fea_train:
        :param fea_test:
        :return:
        """
        sample_data = np.concatenate((fea_train, fea_test))
        attr = self.compute_similarity_before(sample_data)
        return attr

    def predict_class(self, XTest):
        """
        使用(宋祥鑫)样本相似度去做度量来计算测试样本和训练样本之间的距离
        :param XTest:
        :return:
        """
        finalOutput = []
        # data = np.concatenate((self.X, XTest))
        # attr = self.compute_similarity_before(data)
        attr = self.calculate_similarity(self.X, XTest)
        train_size = len(self.X)
        for i in range(train_size, len(data)):         # 测试样本
            d = []
            votes = []
            for j in range(len(self.X)):    # 训练样本
                # strength = (self.massX[j]) * attr[j][i]
                strength = attr[j][i]
                d.append([strength, j])
            d.sort(key=lambda tup: tup[0], reverse=True)
            d = d[0:self.k]
            for distance, j in d:
                votes.append(self.Y[j])
            answer = Counter(votes).most_common(1)[0][0]
            finalOutput.append(answer)
        return np.array(finalOutput)

    def count_distance(self, X_train, X_test, attribute_weights):
        train_length = len(X_train)
        distance = [[0]*len(X_test) for _ in range(train_length)]
        for i in range(train_length):
            x_train_single = X_train[i]
            sign = 0
            for x_test_single in X_test:
                distance[i][sign] = np.sqrt(np.sum(np.square(x_train_single - x_test_single) * attribute_weights))
                # distance[i][sign] = np.sqrt(np.sum(np.square(x_train_single - x_test_single)))
                sign += 1
        return np.array(distance)

    def compute_similarity_before(self, data_no_label):
        num_cores = int(mp.cpu_count()) - 1
        pool = mp.Pool(num_cores)
        ori_data_var = [1, 1]
        modified_sigma = 1
        attribute_number = data_no_label.shape[1]
        [cluster_result, cluster_centers] = self.K_means_complish(data_no_label, 1)
        same_class_cluster, feature_dict_info = self.union_function(data_no_label, cluster_result)
        class_var = self.compute_class_var(same_class_cluster, cluster_centers)
        average_index = int((attribute_number / num_cores) + 1)
        data_no_label_dict = {}
        for i in range(num_cores):
            data_no_label_dict["task{}".format(i + 1)] = [data_no_label[:, average_index * i:average_index * (i + 1)], feature_dict_info, class_var, [i for i in range(average_index * i, (i + 1) * average_index)]]
        results = [pool.apply_async(self.compute_similarity, args=(name, value_information[0], value_information[1], value_information[2], value_information[3], ori_data_var, modified_sigma)) for name, value_information in data_no_label_dict.items()]
        results_1 = [p.get() for p in results]
        similarity_dict = dict()
        for task in results_1:
            for key in task:
                times = int(key[4:]) - 1
                for s in task[key]:
                    similarity_dict[average_index * times + s] = task[key][s]
        similarity_matrix = []
        for key, value in similarity_dict.items():
            similarity_matrix.append(value)
        res_arr = similarity_matrix[0]
        for i in range(1, attribute_number):
            feature_similarity = similarity_matrix[i]
            for position_x in range(feature_similarity.shape[0]):
                for position_y in range(feature_similarity.shape[0]):
                    res_arr[position_x][position_y] = self.t_norm(feature_similarity[position_x][position_y], res_arr[position_x][position_y])
        return res_arr

    def t_norm(self, x, y):
        return min(x, y)

    def K_means_complish(self,dataset, number):
        estimator = KMeans(n_clusters=number, random_state=1)  # 构造聚类器np.transpose
        estimator.fit(np.transpose(dataset))  # 聚类
        label_pred = estimator.labels_  # 获取聚类标签
        centers = estimator.cluster_centers_  # (np.transpose(zero_one_dataset))
        return label_pred, centers

    def union_function(self, ori_data, cluster_result):
        class_info = set(cluster_result)
        res_dict = dict()
        feature_dict = dict()
        for i in class_info:
            res_dict[i] = np.zeros((ori_data.shape[0], 1))
        for i in range(len(cluster_result)):
            res_dict[cluster_result[i]] = np.hstack((res_dict[cluster_result[i]], ori_data[:, i:i + 1]))
            feature_dict[i] = cluster_result[i]
        for key in res_dict:
            res_dict[key] = res_dict[key][:, 1:]
        # print(res_dict)
        return res_dict, feature_dict

    def compute_class_var(self, cluster_dict, center):
        current_class_var = np.zeros((center.shape[0], 1))

        for i in range(current_class_var.shape[0]):
            a = center[i, :]
            current_class_var[i][0] = np.std(np.transpose(a), ddof=0)
        return current_class_var

    def compute_similarity(self, name, data, corr_feature_dict, corr_class_var, feature_position, para1, para2):
        '''0
        计算样本相似性矩阵
        :param data: 同一属性下不同样本属性值（列向量）
        :return: 相似性矩阵
        '''
        res = dict()  # 建立一个字典
        feature_num = data.shape[1]  # 计算特征的数目
        for feature in range(feature_num):
            recent_array = np.zeros((data.shape[0], data.shape[0]))
            recent_feature = data[:, feature]  # 当前需要计算的一列特征
            # print("123456789", max(recent_feature), min(recent_feature))
            recent_sigma = np.std(data[:, feature], ddof=1)
            # print(recent_sigma)
            recent_sigma_yita = 1 - recent_sigma * 0.7
            (yita, index) = self.disjust_ratio(feature_position[feature], corr_feature_dict, corr_class_var,recent_sigma_yita)
            # print(recent_sigma, recent_sigma_yita)
            for x_position in range(data.shape[0]):
                for y_position in range(data.shape[0]):
                    recent_array[x_position][y_position] = self.sample_similarity(recent_feature[x_position],recent_feature[y_position],recent_sigma, yita, index, para2)
            res[feature] = recent_array  # 将计算后的相似度存在于一个字典中，并用属性进行标注 （属性+相似度）
        return {name: res}

    def disjust_ratio(self, corr_feature, corr_feature_dict, corr_class_var, para1):
        attribute_cluster_class = corr_feature_dict[corr_feature]
        attribute_cluster_var = corr_class_var[attribute_cluster_class][0]
        yita = para1
        d_index = 1
        return yita, d_index


    def sample_similarity(self, x, y, sigma, yita, d_index, para2):
        '''
        样本内相似性的计算
        :param x: 样本的参数
        :param y: 样本的参数
        :param sigma: 样本的标准差
        :return: 样本内不同样本相似性
        '''
        # return (math.exp(-((x - y) ** 2) / (2 * (sigma *para2) ** 2)))
        return ((yita * math.exp(-((x - y) ** 2) / (2 * (para2 * sigma) ** 2))) + (1 - yita) * 1) ** d_index




