# -*- coding: utf-8 -*-
# @Time : 2023/12/18 21:00
# @Author : liuhunck
# @File : train.py
import click
import shutil
import os.path

import argparse
import torchvision

import numpy as np

from Model import Model
from model.mnist import mnist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def one_hot(x):
    y = np.zeros((10, ))
    y[x] = 1.
    return y


def transformer(x):
    x = torchvision.transforms.PILToTensor()(x).float()
    return torchvision.transforms.Normalize((0.1307, ), (0.3801, ))(x)


def target_transformer(x):
    return one_hot(x)


def load_data(root, batch_size=0, test_batch_size=0):
    train_data = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                            transform=transformer, target_transform=target_transformer)
    test_data = torchvision.datasets.MNIST(root=root, train=False, download=True,
                                           transform=transformer, target_transform=target_transformer)
    if batch_size == 0:
        batch_size = len(train_data)
    if test_batch_size == 0:
        test_batch_size = len(test_data)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=True)
    print(f"成功加载训练集，大小为：{len(train_data)}")
    print(f"成功加载测试集，大小为：{len(test_data)}")
    return train_loader, test_loader


def test(network: Model, loss_fn, test_loader, train_counter):
    test_acc = 0
    test_acc_size = 0
    tot_test_loss = 0
    for data in test_loader:
        images, targets = data
        images = images.numpy()
        targets = targets.numpy()
        predict = network(images)
        test_loss = loss_fn(predict, targets)
        tot_test_loss += test_loss
        test_acc += np.sum(np.argmax(predict, axis=1) == np.argmax(targets, axis=1))
        test_acc_size += len(targets)

    test_acc /= test_acc_size
    test_loss = tot_test_loss / len(test_loader)
    print(f"测试次数: {train_counter / 10}, Loss = {test_loss}, Acc = {test_acc}")
    writer.add_scalars("train", {"test_loss": test_loss, "test_acc": test_acc}, train_counter)


def train(network: Model, loss_fn, lr, train_loader, test_loader, max_epoch):
    train_counter = 0
    train_acc = 0
    train_acc_size = 0
    for epoch in range(max_epoch):
        print(f"---------第 {epoch} 轮训练开始---------")

        for data in train_loader:
            images, targets = data
            images = images.numpy()
            targets = targets.numpy()
            output = network(images)
            train_loss = loss_fn(output, targets)
            network.backward(output - targets)
            network.update(lr)
            train_counter += 1

            train_acc += np.sum(np.argmax(output, axis=1) == np.argmax(targets, axis=1))
            train_acc_size += len(targets)

            if train_counter % 100 == 0:
                train_acc /= train_acc_size
                print(f"训练次数: {train_counter}, Loss = {train_loss}, Acc = {train_acc}")
                writer.add_scalars("train", {"train_loss": train_loss, "train_acc": train_acc}, train_counter)
                train_acc = 0
                train_acc_size = 0

            if train_counter % 1000 == 0:
                test(network, loss_fn, test_loader, train_counter)


def cross_entropy_loss(y_hat, y):
    m = y.shape[0]
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--batch', default=64, type=int, help='Batch size')
    parser.add_argument('--max-epoch', default=5, type=int, help='Max training epoch')
    parser.add_argument('-o', '--output', default='./model/mnist.npt', type=str,
                        help='The output filename of trained weight and bias')
    parser.add_argument('--log-dir', default='./logs', type=str, help='The log-dir of tensorboard')
    args = parser.parse_args()
    log_dir = args.log_dir
    if os.path.exists(log_dir):
        if click.confirm(f'The folder {log_dir} has exists, delete it? ', default=False):
            shutil.rmtree(log_dir)
            print(f'The folder {log_dir} was deleted...')
        writer = SummaryWriter(log_dir)

    net = Model(mnist)
    train_loader, test_loader = load_data('./data', args.batch)
    train(net, cross_entropy_loss, args.lr, train_loader, test_loader, args.max_epoch)

    net.save(args.output)

