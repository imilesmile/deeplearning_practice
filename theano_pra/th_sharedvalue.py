#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
theano de shared value
"""

import numpy as np
import theano.tensor as T
import theano

#共享变量中数据类型很重要,定义vector和matrix时需要统一
#最后一个参数是共享变量的名称
#初始化共享变量state为0
state = theano.shared(np.array(0, dtype = np.float64), 'state')

#定义累加值, 名称为inc, 定义数据类型是需要用state.dtype,而不是dtype = np.float64
#否则会报错
inc = T.scalar('inc', dtype = state.dtype)
#定义一个accumulator函数
#输入为inc, 输出为state
#累加的过程叫做updates, 作用是state = state+inc
accumulator = theano.function([inc], state, updates=[(state, state + inc)])

"""
get_value， set_value 这两种只能在 Shared 变量 的时候调用。
"""

#获取共享变量的值get_value
print(state.get_value())
#output: 0.0

accumulator(1)
print(state.get_value())
#output: 1.0

accumulator(10)
print(state.get_value())
#output: 11.0

#设置共享变量的值set_value
state.set_value(-1)
accumulator(3)
print(state.get_value())
#output : 2.0



#临时试用共享变量
#有时需要暂时试用shared变量,不需要更新,这时可以定义一个临时变量代替共享变量
#函数输入值为[Inc, a], 要用a带入state, 输出是tmp_func函数形式
#givens表示需要把什么替换成什么, 而state不会改变
#最后输出结果中, state暂时被替换成a,state值不会变,还是上步的值2, a的值是3
a = T.scalar(dtype=state.dtype)
tmp_func = state * 2 + inc
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)])
print(skip_shared(2, 3))
#output: 3 *2+2 = 8


















