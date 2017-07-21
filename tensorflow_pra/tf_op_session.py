#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:56:35 2017

@author: miao

创建一个常量op 产生一个1x2的矩阵, 这个op是添加在默认图中的一个节点
 op 构造函数返回代表已被组织好的 op 作为输出对象，这些对象可 以传递给其它 op 构造函数作为输入。
"""
import tensorflow as tf

#通过构造器返回一个输出常量
matrix1 = tf.constant([[3.,3.]])

#创建一个2X1的矩阵
matrix2 = tf.constant([[2.],[2.]])

#创建一个Matmul,用matrix1和matrix2作为输入,返回两个矩阵相乘的结果
product = tf.matmul(matrix1,matrix2)

#默认图现在拥有三个节点，两个constant() op，一个matmul() op. 
#为了真正进行矩 阵乘法运算，得到乘法结果, 你必须在一个会话 (session) 中载入动这个图。
#创建一个 会话对象 (Session object)。会话构建器在未指明参数时会载入默认的图。
sess = tf.Session()

#要运行matmul op 可以运行run()方法
result = sess.run(product)
print result

#关闭会话
sess.close()

"""
会话在完成后必须关闭以释放资源。你也可以使用"with"句块开始一个会话，该会
话将在"with"句块结束时自动关闭。   
  
with tf.Session() as sess: result = sess.run([product])
  print(result)
"""

