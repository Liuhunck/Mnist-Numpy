# -*- coding: utf-8 -*-
# @Time : 2024/2/18 21:28
# @Author : liuhu
# @File : test.py
import torch
from torch.nn import BatchNorm2d as BN
import numpy as np
from Layer import BatchNorm2d

inp = np.random.randn(4, 3, 3, 3) * 0.1 + 5
out = np.random.randn(4, 3, 3, 3) * 10

bn = BatchNorm2d(3)
oup = bn.forward(inp, is_training=True)
dx = bn.backward(out)

inp1 = torch.tensor(inp, requires_grad=True)
w = torch.tensor(out)
bn1 = BN(3, dtype=torch.float64)
bn1.train(True)
oup1 = bn1.forward(inp1)
oo = (oup1 * w).sum()
oo.backward()

print(bn.dw)
print(bn1.weight.grad)
