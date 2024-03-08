import numpy as np
import matplotlib.pyplot as plt

x_train = np.random.rand(200,1)
y_train = 3 + 5 * x_train + np.random.normal(0,1,size=(200,1))

plt.scatter(x_train,y_train)

theta = np.zeros((2,1))
x_train = np.hstack((np.ones((200,1)),x_train))
for _ in range(5000):
    theta = theta - 0.01*1/200 * np.dot(((np.dot(x_train,theta) - y_train).T),x_train).T

x_predict = x_train
y_predict = np.dot(x_predict,theta)

plt.plot(x_predict,y_predict,'r')
plt.xlim(0,1)
plt.ylim(0,8)
plt.show()

print(theta)