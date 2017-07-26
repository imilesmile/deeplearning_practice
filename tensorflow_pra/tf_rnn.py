#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tensorflow之RNN
循环神经网络做手写数据集分类
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#设置随机数来比较两种计算结果
tf.set_random_seed(1)

#导入手写数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#设置参数
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST 输入为图片(img shape: 28*28)对应到图片像素的一行
n_steps = 28    # time steps 对应到图片有多少列
n_hidden_units = 128   # 隐藏层神经元个数
n_classes = 10      # MNIST分类结果为10

#定义权重
weights = {
        #(28,128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
        #(128,10)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
#定义bias
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    #作为cell输入的隐藏层
    ######################################################
    #输入层
    #将输入shape从X三维输入变为二维(128 batch * 28 steps, 128 hidden)
    X = tf.reshape(X, [-1,n_inputs])
    
    #隐藏层
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 传给cell时需要将二维转为三维X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    #cell
    #######################################################
    #LSTM cell forget_bias=1.0表示最开始学习我们不希望忘掉任何state, 
　　 #state_is_tuple=True这个为true表示记录每个时间点的cell状态和输出值,以后会默认为true
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #将lstm cell 分成两部分(c_state, h_state)，对应到lstm一个是主线c_state（没有cell的遗忘），
　　 #支线是h_state（有cell的遗忘），zero_state将每个t时间的cell初始化为0，
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    #outputs为lstm所有输出结果包括每个时刻cell的state，和输出值，final_state为最后的结果，
　　 #time_major参数表示时间序列的位置是否为输入数据的第一个维度，由于我们是在第二个维度，所以为false
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    
    #1.将隐藏层的输出作为最后结果，只有一个结果
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    
    #2.将每一步的结果输出到lists，在对outputs unstack后[1,0, 2]是将outputs list中每个tuple中元素对应展开
    tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out'] # shape = (128, 10)
    
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1