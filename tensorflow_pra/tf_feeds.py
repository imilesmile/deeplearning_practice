#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:56:14 2017

@author: miao
"""
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #feed_dict是字典形式传值
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    
 # output:
# [array([ 14.], dtype=float32)]