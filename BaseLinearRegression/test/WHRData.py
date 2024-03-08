import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

whr = pd.read_csv("../data/world-happiness-report-2017.csv")
input_param_name = ['Economy..GDP.per.Capita.','Health..Life.Expectancy.']
output_param_name = 'Happiness.Score'

input_data = whr[input_param_name].values
output_data = whr[[output_param_name]].values

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter3D(input_data[:,0], input_data[:,1], output_data[:,0], color="green")
# ax.set_xlabel('Economy')
# ax.set_ylabel('Health')
# ax.set_zlabel('Happiness')
# plt.show()


def normalize(features):
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation

features_normalized, features_mean, features_deviation = normalize(input_data)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(features_normalized[:,0], features_normalized[:,1], output_data[:,0], color="green")
ax.set_xlabel('Economy')
ax.set_ylabel('Health')
ax.set_zlabel('Happiness')
plt.show()