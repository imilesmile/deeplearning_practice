#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tensorflow dropout
dropout一般用在全连接的部分，卷积部分不会用到dropout,输出曾也不会使用dropout，适用范围[输入，输出)
只用在训练集,不用在测试集
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#加载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, keep_prob=1.0):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    # 这里做 dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

#添加输出层
l1 = add_layer(xs, 64 ,50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
                                              
#scalar_summary记录存数值,用于画图
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#定义每次做连接的神经元个数
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()
#
#在TensorFlow中，所有的操作只有当你执行，或者另一个操作依赖于它的输出时才会运行。
#我们刚才创建的这些节点（summary nodes）都围绕着你的图像：没有任何操作依赖于它们的结果。
#因此，为了生成汇总信息，我们需要运行所有这些节点。这样的手动工作是很乏味的，
#因此可以使用tf.merge_all_summaries来将他们合并为一个操作。
#http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/summaries_and_tensorboard.html
merged = tf.summary.merge_all()

#summary
train_writer = tf.train.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.train..summary.FileWriter("logs/test", sess.graph)


sess.run(tf.initialize_all_variables())

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if i % 50 == 0:
        # record loss
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)












