# -*- coding: utf-8 -*-
"""
Author:liuhunck
LANG:Python
TASK:Model
"""

import pickle
from Layer import Dropout


class Model:
    def __init__(self, net):
        self.net = net
        self.is_training = False

    def __call__(self, x):
        return self.forward(x)

    def train(self, is_training=True):
        self.is_training = is_training

    def forward(self, x):
        for i in self.net:
            if isinstance(i, Dropout):
                x = i.forward(x, is_training=self.is_training)
            else:
                x = i.forward(x)
        return x

    def backward(self, y):
        for i in reversed(self.net):
            y = i.backward(y)

    def update(self, alpha):
        for i in self.net:
            i.update(alpha)

    def save(self, path):
        para = []
        for i in self.net:
            para.append(i.get_para())
        with open(path, "wb") as f:
            f.write(pickle.dumps(para))

    def load(self, path):
        with open(path, "rb") as f:
            para = pickle.loads(f.read())
        for i in range(len(self.net)):
            self.net[i].load_para(para[i])

