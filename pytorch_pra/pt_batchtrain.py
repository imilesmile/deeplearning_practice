#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data

torch.manual_seed(1)

#批训练数据的个数
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

#先转换成torch为dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

#把dataset放入DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    #将数据打乱
    shuffle=True,
    #多数据读取数据
    num_workers=2
)

#训练所有数据3次
for epoch in range(3):
    #每一步loader使用一小批数据
    for step, (batch_x, batch_y) in enumerate(loader):
        print 'Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy()