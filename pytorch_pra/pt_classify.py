#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#创建数据
n_data = torch.ones(100,2)
#类型0
x0 = torch.normal(2*n_data, 1) #x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)          #y data (tensor), shape=(100, 1)
#类型1
x1 = torch.normal(-2*n_data, 1)#x data (tensor), shape=(100, 1)
y1 = torch.zeros(100)          #y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)# FloatTensor = 32-bit floating
y = torch.cat((y0,y1),).type(torch.LongTensor)    # LongTensor = 64-bit integer

#torch 只能在Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = y.data.numpy(),
            s=100, lw=0, cmap='RdYlGn')
# plt.show()

#建立神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden =  torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)
print net
"""
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
"""

#训练网络
#传入net的所有参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
#算误差时, 真实值不是one-hot 形式, 而是1D Tensor, (batch,)
#但是预测值是2D tensor(batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()


plt.ion()
plt.show()

for t in range(100):
    #放入训练数据x, 输出前向分析值
    out = net(x)
    #计算误差
    loss = loss_func(out, y)

    #清空上一步残余更新的参数值
    optimizer.zero_grad()
    #误差反向传播, 计算参数更新值
    loss.backward()
    #将参数更新值施加到net的parameteres
    optimizer.step()

    if t % 2 ==0:
        plt.cla()
        #softmax输出最大概率是预测值
        prediction = torch.max(F.softmax(out),1)[1]
        #np.squeeze() 可以直接进行某一维度维度压缩
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:, 1], c=pred_y, s = 100, lw = 0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5, -4, 'Accuracy=%.2f'%accuracy, fontdict={'size': 20, 'color': 'red'})
    plt.ioff()#停止画图
    plt.show()






