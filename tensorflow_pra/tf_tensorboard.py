#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tensorboard 可视化生成tensorboard文件
"""
import tensorflow as tf


def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    #添加一层名称定义
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            #将weights添加到histogram标签下
            tf.hi
    
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

"""
tensorboard文件生成的位置,文件位置为../logs
"""
writer = tf.summary.FileWriter("../logs/", sess.graph)

# important step
sess.run(tf.initialize_all_variables())
