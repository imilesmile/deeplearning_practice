#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
theano基础
"""

import numpy as np
import theano.tensor as T
from theano import function

#定义X和Y两个常量 (scalar)，把结构建立好之后，把结构放在function，在把数据放在function。
# 建立 x 的容器
x = T.dscalar('x')
# 建立 y 的容器
y = T.dscalar('y') 
#  建立方程
z = x + y

# 使用 function 定义 theano 的方程
# 将输入值 x, y 放在 [] 里,  输出值 z 放在后面
f = function([x, y], z)

# 将确切的 x, y 值放入方程中
print(f(2, 3))

# output: 5.0

#theano 中 的 pp (pretty-print) 能够打印出原始方程:
from theano import pp
print(pp(z))
# output:(x + y)

#定义矩阵，以及利用矩阵做相关运算:
# 矩阵 x 的容器
x = T.dmatrix('x')
# 矩阵 y 的容器
y = T.dmatrix('y')
# 定义矩阵加法
z = x + y
# 定义方程
f = function([x, y], z)
print(f(
        np.arange(12).reshape(3,4),#x为3行4列的矩阵
        10 * np.ones((3, 4)) #y也为3行4列,且数值全为10
        ))

"""
output:
[[ 10.  11.  12.  13.]
 [ 14.  15.  16.  17.]
 [ 18.  19.  20.  21.]]
"""