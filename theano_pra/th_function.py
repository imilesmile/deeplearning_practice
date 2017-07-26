#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
theano之function
"""

import numpy as np
import theano.tensor as T
import theano

#首先需要定义一个 tensor T
x = T.dmatrix('x')
#声明sigmoid激活函数函数
s = 1/(1+T.exp(-x))
#调用 theano 定义的计算函数 logistic
logistic = theano.function([x], s)
#输入为一个2行两列的矩阵
print(logistic([[0, 1], [-2, -3]]))



#多输入,多输出的函数
#如输入值为两个,输出值为两个
#输入的值为矩阵A, B
a, b = T.dmatrices('a', 'b')
#计算a, b之间的diff, abs_diff, diff_squared
diff = a-b
abs_diff = abs(diff)
diff_squared = diff**2
#定义多输出函数
f = theano.function([a, b ], [diff, abs_diff, diff_squared])

x1, x2, x3 = f(
        np.ones((2, 2)),
        np.arange(4).reshape((2,2))
        )
print(x1, x2, x3)
"""
output:
    (array([[ 1.,  0.],
       [-1., -2.]]), array([[ 1.,  0.],
       [ 1.,  2.]]), array([[ 1.,  0.],
       [ 1.,  4.]]))
"""

#使用 T.dscalars() 里面同时定义三个纯量的容器。 以及输出值z
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
#定义theano函数
f = theano.function(
        [x,
         theano.In(y, value=1),
         theano.In(w, value=2, name='weights')],
        z
        )
#使用默认值
print(f(23))
#使用非默认值
print(f(23, 1, 4))
#试用名称赋值
print(f(23,1, weights = 6))

"""
output:
    48.0
    96.0
    144.0
"""