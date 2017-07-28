#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2],[3, 4]])
#在BP的时候，pytorch是将Variable的梯度放在Variable对象中的，
# 我们随时都可以使用Variable.grad得到对应Variable的grad。
# 刚创建Variable的时候，它的grad属性是初始化为0.0的。
#需要求导的话，requires_grad=True属性是必须的。
variable = Variable(tensor, requires_grad = True)
print tensor
"""
output:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
#多了一个Variable containing:
print variable
"""
output:
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

#对比tensor的计算和variable的计算
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print t_out
"""
7.5
"""
print v_out
"""
Variable containing:
 7.5000
 [torch.FloatTensor of size 1]
"""

#模拟v_out 反向误差
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2
v_out.backward()

print variable.grad

#获取variable的数据
print variable
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
print variable.data
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
print variable.data.numpy()
"""
[[ 1.  2.]
 [ 3.  4.]]
 """