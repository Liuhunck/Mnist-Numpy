# -*- coding: utf-8 -*-
"""
Author:liuhunck
LANG:Python
TASK:Layer
"""
import numpy as np


# 层的基类
class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, y):
        pass

    def update(self, alpha):
        pass

    def get_para(self):
        return ()

    def load_para(self, para):
        pass


# 全连接层
class Linear(Layer):
    def __init__(self, n, m, sigma=0.01):
        super().__init__()
        self.w = np.random.randn(n, m) * sigma
        self.b = np.zeros((1, m))
        self.dw = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, y):
        self.dw = np.dot(self.x.T, y) / y.shape[0]
        self.db = np.sum(y, axis=0, keepdims=True) / y.shape[0]
        return np.dot(y, self.w.T)

    def update(self, alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db

    def get_para(self):
        return self.w, self.b

    def load_para(self, para):
        self.w, self.b = para


# Softmax 层（包含全连接层和激活函数）
class Softmax(Layer):
    def __init__(self, n, m, sigma=0.01):
        super().__init__()
        self.w = np.random.randn(n, m) * sigma
        self.b = np.zeros((1, m))
        self.dw = None
        self.db = None
        self.x = None

    @staticmethod
    def softmax(x):
        exp_z = np.exp(x)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        self.x = x
        x = np.dot(x, self.w) + self.b
        x = self.softmax(x)
        return x

    def backward(self, y):
        self.dw = np.dot(self.x.T, y) / y.shape[0]
        self.db = np.sum(y, axis=0, keepdims=True) / y.shape[0]
        return np.dot(y, self.w.T)

    def update(self, alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db

    def get_para(self):
        return self.w, self.b

    def load_para(self, para):
        self.w, self.b = para


# 摊平层
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, x):
        self.s = x.shape
        return np.reshape(x, (x.shape[0], -1))

    def backward(self, y):
        return np.reshape(y, self.s)


# Sigmoid 激活函数层
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def backward(self, y):
        return np.multiply(y, np.multiply(self.x, 1 - self.x))


# ReLU 激活函数层
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x > 0
        return np.maximum(x, 0)

    def backward(self, y):
        return np.multiply(self.x, y)


# 2维卷积层
class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, sigma=0.01):
        """
        卷积层构造函数 数据格式CHW
        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        :param kernel_size: 卷积核大小，可以是一个整数或一个元组（height, width）
        :param stride: 步幅大小，可以是一个整数或一个元组（height, width），默认为1
        :param padding: 填充大小，可以是一个整数或一个元组（height, width），默认为0
        :param sigma: 初始化标准差
        """
        super().__init__()

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.w = np.random.randn(input_channels * self.kernel_size[0] * self.kernel_size[1], output_channels) * sigma
        self.b = np.zeros((1, output_channels, 1, 1))
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.x = None

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入数据，形状为 (batch_size, input_channel, input_height, input_width)
        :return: 卷积结果，形状为 (batch_size, output_channel, output_height, output_width)
        """
        self.x = x

        output_channels = self.w.shape[1]
        kernel_height, kernel_width = self.kernel_size
        batch_size, input_channels, input_height, input_width = x.shape

        output_height = (input_height + 2 * self.padding[0] - kernel_height) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] - kernel_width) // self.stride[1] + 1

        x = np.pad(x, [(0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])],
                   mode="constant")

        y = np.zeros((batch_size, output_channels, output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                hs, ws = h * self.stride[0], w * self.stride[1]
                he, we = hs + kernel_height, ws + kernel_width
                y[:, :, h, w] = np.dot(x[:, :, hs:he, ws:we].reshape((batch_size, -1)), self.w)
        y += self.b
        return y

    def backward(self, y):
        """
        反向传播函数
        :param y: 输出特征图的梯度，形状为 (batch_size, output_channels, output_height, output_width)
        :return: 输入特征图的梯度，形状为 (batch_size, input_channels, input_height, input_width)
        """
        kernel_height, kernel_width = self.kernel_size
        _, input_channels, input_height, input_width = self.x.shape
        batch_size, output_channels, output_height, output_width = y.shape

        self.dw.fill(0.)
        self.db.fill(0.)
        x = np.zeros(self.x.shape)

        kx_size = (batch_size, input_channels, kernel_height, kernel_width)
        for h in range(output_height):
            for w in range(output_width):
                hs, ws = h * self.stride[0], w * self.stride[1]
                he, we = hs + kernel_height, ws + kernel_width
                y_s = y[:, :, h, w]

                x[:, :, hs:he, ws:we] += np.dot(y_s, self.w.T).reshape(kx_size)
                self.dw += np.dot(self.x[:, :, hs:he, ws:we].reshape((batch_size, -1)).T, y_s)
                self.db += np.sum(y_s, axis=0).reshape((1, -1, 1, 1))

        self.dw /= batch_size
        self.db /= batch_size

        return x

    def update(self, alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db

    def get_para(self):
        return self.w, self.b

    def load_para(self, para):
        self.w, self.b = para


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=0, padding=0):
        super().__init__()

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = self.kernel_size if stride == 0 else stride
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.x = None

    def forward(self, x):
        self.x = x

        kernel_height, kernel_width = self.kernel_size
        batch_size, input_channels, input_height, input_width = x.shape

        output_height = (input_height + 2 * self.padding[0] - kernel_height) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] - kernel_width) // self.stride[1] + 1

        x = np.pad(x, [(0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])],
                   mode="constant")

        y = np.zeros((batch_size, input_channels, output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                hs, ws = h * self.stride[0], w * self.stride[1]
                he, we = hs + kernel_height, ws + kernel_width
                y[:, :, h, w] = np.max(x[:, :, hs:he, ws:we], axis=(2, 3))

        return y

    def backward(self, y):

        kernel_height, kernel_width = self.kernel_size
        _, _, output_height, output_width = y.shape

        x = np.zeros(self.x.shape)

        for h in range(output_height):
            for w in range(output_width):
                hs, ws = h * self.stride[0], w * self.stride[1]
                he, we = hs + kernel_height, ws + kernel_width

                x_s = self.x[:, :, hs:he, ws:we]

                x[:, :, hs:he, ws:we] += (x_s == np.max(x_s, axis=(2, 3), keepdims=True)) * y[:, :, h:h+1, w:w+1]

        return x


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, is_training=False):
        if is_training:
            self.mask = (np.random.randn(*x.shape) < self.dropout_rate) / self.dropout_rate
            return np.multiply(x, self.mask)
        return x

    def backward(self, y):
        return np.multiply(y, self.mask)


