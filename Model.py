# -*- coding: utf-8 -*-
"""
Author:liuhunck
LANG:Python
TASK:Model
"""

import pickle


class Model:
    def __init__(self, net):
        self.net = net

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for i in self.net:
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

