#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:56:42 2017

@author: miao
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#构建数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points


#建立模型
#用Sequential 建立model, 再用model.add添加神经层,添加的是dense全连接层
model = Sequential()
#回归的的输入和输出都为1
model.add(Dense(input_dim = 1, output_dim = 1))

#激活模型
#误差用mse, 优化器用随机梯度下降
model.compile(loss='mse', optimizer='sgd')

#训练模型
print('train=========')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 ==0:
        print('train cost: ', cost)
        
"""
train=========
('train cost: ', 4.3885427)
('train cost: ', 0.21306995)
('train cost: ', 0.039464761)
('train cost: ', 0.01153493)
"""       

#测试模型
#model.evaluate，输入测试集的x和y， 输出 cost，weights 和 biases
#其中 weights 和 biases 是取在模型的第一层 model.layers[0] 学习到的参数。
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost: ', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

"""
Testing ------------
40/40 [==============================] - 0s
('test cost: ', 0.011480952613055706)
('Weights=', array([[ 0.33378708]], dtype=float32), 
'biases=', array([ 1.98737764], dtype=float32))
"""

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
