#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
简单的神经网络
"""

import tensorflow as tf
import numpy as np

def add_layer(input_value, in_size, out_size, activate_function=None):
    #权重
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #偏移量
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    #wx+b
    Wx_plus_b = tf.matmul(input_value,Weights)+biases
    #有激活函数则使用激活函数计算下一层的值
    if activate_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    return outputs
    
#创造些简单的数据
#创建一个300行数值在-1到1 之间的一个输入向量
#np.linspace函数可以生成元素为300的等间隔数列
#np.newaxis的功能是插入新维度
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
#添加噪声,模拟真实数据,数据格式和x一样
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise

#定义placeholder作为神经网络的输入
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#添加隐藏层,隐藏层的单元数设为10个,激活函数为relu
l1 = add_layer(xs, 1, 10, activate_function=tf.nn.relu)

#添加输出层,输入数据为最后一层隐藏层的输出,激活函数线性函数
prediction = add_layer(l1, 10, 1, activate_function=None)

#定义损失函数,损失函数为平方损失
#reduction_indices参数，表示函数的处理维度,
#没有reduction_indices这个参数，此时该参数取默认值None，将把input_tensor降到0维，也就是一个数。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=1))

#定义训练的公式,tensorflow每执行一次都在重复执行这个训练步骤, 
#优化方法是梯度下降法,学习速率是0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#导入step
#tensorflow定义图结构中有变量时必须先对变量进行初始化,激活变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#训练迭代1000次
for i in range(1000):
    #训练
    sess.run(train_step, feed_dict={xs : x_data, ys : y_data})
    #每50步输出一下损失函数的优化情况
    if i % 50 ==0:
        print sess.run(loss, feed_dict={xs : x_data, ys : y_data})

#输出结果:
    #0.257926
    #0.0173966
    #0.00949054
    #0.00592314
    #0.00461547
    #0.00421694
    #0.00411316
    #0.00403906
    #0.00396101
    #0.00388605
    #0.00381183
    #0.00374701
    #0.00368595
    #0.00362093
    #0.00354979
    #0.00347736
    #0.00339087
    #0.00331123
    #0.00324889
    #0.00318681  

