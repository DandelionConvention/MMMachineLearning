import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BaseLinearRegression.SimpleLinearRegression import LinearRegression

data = pd.read_csv('./data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

input_param_name = ['Economy..GDP.per.Capita.','Health..Life.Expectancy.']
output_param_name = 'Happiness.Score'

x_train = train_data[input_param_name].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

num_iterations = 500
learning_rate = 0.01


linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)

linear_regression_no_standar = LinearRegression(x_train,y_train,normalize_data=False)
(theta_no_standar,cost_history_no_standar) = linear_regression_no_standar.train(learning_rate,num_iterations)

plt.plot(range(num_iterations),cost_history,'r')
plt.plot(range(num_iterations),cost_history_no_standar,'g')
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()
