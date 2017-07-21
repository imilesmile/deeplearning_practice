#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:35:43 2017

@author: miao
使用变量实现一个 简单的计数器
"""

import tensorflow as tf

#建立一个变量,用0初始化它的值
state = tf.Variable(0, name= "counter")

#创建一个op one
one = tf.constant(1)
new_value = tf.add(state, one)
#代码中assign()操作是图所描绘的表达式的一部分, 正如add()操作一样. 所以在调 用run()执行表达式之前, 它并不会真正执行赋值操作.
update = tf.assign(state, new_value)

#变量在启动图计算之后必须通过运行'init'来初始化
init_op = tf.initialize_all_variables()

#启动图运行ops
with tf.Session() as sess:
    #先初始化'init'op
    sess.run(init_op)
    #打印state初始化值
    print sess.run(state)
    #运行update
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
#输出:
    0
    1
    2
    3