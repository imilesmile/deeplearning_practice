#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tensorflow做分类
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#手写数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义神经层
def add_layer(inputs, in_size, out_size, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# 定义输入 placeholder
xs = tf.placeholder(tf.float32, [None, 784])#图片的像素为28X28
ys = tf.placeholder(tf.float32, [None, 10])


#添加输出层,隐藏层有10个神经元,softmax做激活函数
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 定义损失函数为交叉熵损失
#reduction_indices是指沿tensor的哪些维度求和。

#'x' is [[1, 1, 1]
#         [1, 1, 1]]
#tf.reduce_sum(x) ==> 6
#tf.reduce_sum(x, 0) ==> [2, 2, 2]
#tf.reduce_sum(x, 1) ==> [3, 3]
#tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], ]3]]
#tf.reduce_sum(x) ==> 6
#tf.reduce_sum(x) ==> 6
#tf.reduce_sum(x, [0, 1]) ==> 6
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

#学习速率为0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    #随机批梯度下降
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
         print(compute_accuracy(mnist.test.images, mnist.test.labels))