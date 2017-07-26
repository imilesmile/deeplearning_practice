#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
theano de classify
"""

import numpy as np
import theano.tensor as T
import theano

#计算分类准确率
def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy


#训练数据的个数
N = 400
#训练数据的特征数
feats = 784

#生成随机数
D = (np.random.randn(N,feats), np.random.randint(size = N, low = 0, high =2))

print(D)

#构建神经网络
#定义x y容器, 相当于tensorflow中的placeholder
x = T.dmatrix("x")
y = T.dvector("y")

#初始化weights和bias, weights的数量和features一样

W = theano.shared(np.random.randn(feats), name =  'w')
b = theano.shared(0., name='b')

#定义激活函数(sigmoid), 加入l1正则化
p_1 = T.nnet.sigmoid(T.dot(x,W) + b)
#sigmoid值大于0.5为true
prediction = p_1 > 0.5
#定义交叉熵损失函数
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) 

#加入l2正则化,减少过拟合
cost = xent.mean() + 0.01 * (W**2).sum()
#定义梯度迭代的gW, gb,用于更新参数
gW, gb = T.grad(cost, [W, b])

#激活神经网络
learning_rate = 0.1
train = theano.function(
        inputs = [x, y],
        outputs = [prediction, xent.mean()],
        updates = ((W, W - learning_rate* gW), (b, b - learning_rate * gb))
        )

predict = theano.function(inputs = [x], outputs = prediction)

#训练模型
for i in range(500):
    pred, err = train(D[0], D[1])
    if i % 50 ==0:
        print('cost', err)
        print('accuracy', compute_accuracy(D[1], predict(D[0])))

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))


























