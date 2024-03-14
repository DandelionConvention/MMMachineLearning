import numpy as np
from utils.features import prepare_for_training

class LinearRegression:

    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
        # polynomial_degree: 多项式特征的最高次数，默认为0，表示不使用多项式特征。
        #
        # sinusoid_degree: 正弦函数特征的最高次数，默认为0，表示不使用正弦函数特征。

        self.data = data_processed
        # (n,2)多补了一列1
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1] # 特征个数
        self.theta = np.zeros((num_features,1))

    def train(self,alpha,num_iterations = 1000):
        """
        训练入口
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return: 
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history


    def gradient_descent(self,alpha,num_iterations):
        """
        训练模块，执行梯度下降
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return:
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        """
        梯度下降计算，矩阵运算
        :param alpha: 学习率
        :return:
        """
        num_examples = self.data.shape[0] # 一组多少个
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        self.theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T

    def cost_function(self,data,labels):
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

    @staticmethod
    def hypothesis(data,theta):
        return np.dot(data,theta)

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        return self.cost_function(data_processed,labels)

    def predict(self,data):
        """
        预测模块
        :param data: 数据
        :return:
        """
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        prediction = LinearRegression.hypothesis(data_processed,self.theta)
        return prediction