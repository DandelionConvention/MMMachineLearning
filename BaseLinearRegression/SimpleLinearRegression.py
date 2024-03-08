import numpy as np
from BaseLinearRegression.utils.features import normalize

class LinearRegression:
    def __init__(self,data,labels,normalize_data=True):
        # 计算样本总数
        num_examples = data.shape[0]
        data_processed = np.copy(data)
        self.normalize_data = normalize_data
        # 预处理
        data_normalized = data_processed
        if normalize_data:
            (
                data_normalized,
                features_mean,
                features_deviation
            ) = normalize(data_processed)

            data_processed = data_normalized
        # 加一列1
        data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

        self.data = data_processed
        self.labels = labels
        num_features = self.data.shape[1] # 特征个数
        self.theta = np.zeros((num_features,1))

    def train(self,alpha,num_iterations = 500):
        loss_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,loss_history

    def gradient_descent(self,alpha,num_iterations):
        loss_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            loss_history.append(self.loss_function(self.data,self.labels))
        return loss_history

    def gradient_step(self, alpha):
        num_examples = self.data.shape[0] # 一组多少个
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        self.theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
    @staticmethod
    def hypothesis(data, theta):
        return np.dot(data, theta)

    def loss_function(self,data,labels):
        """
        损失计算
        :param data: 数据
        :param labels: 标签
        :return:
        """
        num_examples = self.data.shape[0]  # 一组多少个
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/num_examples
        return cost[0][0]
    def get_cost(self,data,labels):
        # data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        if self.normalize_data:
            (
                data_normalized,
                features_mean,
                features_deviation
            ) = normalize(data)

            data_processed = data_normalized
        # 加一列1
        data_processed = np.hstack((np.ones((data.shape[0], 1)), data_processed))
        return self.loss_function(data_processed,labels)

    def predict(self,data):
        """
        预测模块
        :param data: 数据
        :return:
        """
        if self.normalize_data:
            (
                data_normalized,
                features_mean,
                features_deviation
            ) = normalize(data)

            data_processed = data_normalized
        # 加一列1
        data_processed = np.hstack((np.ones((data.shape[0], 1)), data_processed))
        prediction = LinearRegression.hypothesis(data_processed,self.theta)
        return prediction