# -*- coding: utf-8 -*-
# @Time : 2024/2/18 21:28
# @Author : liuhu
# @File : test.py

import PIL.Image
import torchvision

root = './data'
test_data = torchvision.datasets.MNIST(root=root, train=False, download=True)

cnt = 0
for img, tar in test_data:
    PIL.Image.Image.save(img, './images/%d-%d.png' % (cnt, tar))
    cnt += 1
    if cnt == 16:
        break

