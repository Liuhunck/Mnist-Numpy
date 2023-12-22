# -*- coding: utf-8 -*-
# @Time : 2023/12/22 10:16
# @Author : liuhu
# @File : run.py

import argparse
import os.path

import PIL.Image
import numpy as np

from Model import Model
from model.mnist import mnist
from train import transformer


def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    ext = os.path.splitext(file_path)[1].lower()
    if ext in image_extensions:
        return True
    else:
        return False


def load_image(file_path):
    try:
        _ = PIL.Image.open(file_path).convert('L').resize((28, 28))
        return True, os.path.basename(file_path), _
    except Exception:
        return False, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand-written digits recognition')
    parser.add_argument('--model-para', type=str, default='./model/mnist.npt', help='The model wights')
    parser.add_argument('--input', type=str, required=True, help='The image of input')
    parser.add_argument('--output', type=str, default='./run', help='The output dir')
    args = parser.parse_args()

    net = Model(mnist)
    model_para = args.model_para
    if os.path.isfile(model_para):
        net.load(model_para)
    else:
        print(f'Model parameter file {model_para} not found...')
        exit()

    img = args.input
    image_files = []
    image_names = []
    if os.path.isfile(img):
        image_files.append(img)
    elif os.path.isdir(img):
        image_files = []
        for root, dirs, files in os.walk(img):
            for file in files:
                file_path = os.path.join(root, file)
                if is_image_file(file_path):
                    image_files.append(file_path)
    else:
        print(f'{img} not found')
        exit()

    images = []
    imgs = []
    for file in image_files:
        _, name, image = load_image(file)
        if _:
            image_names.append(name)
            images.append(image)
            imgs.append(transformer(image).numpy())
            print(f'Load image {name} success...')
        else:
            print(f'Load image {name} error...')

    out = net(np.array(imgs))
    out = np.argmax(out, axis=1)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    for label, name, image in zip(out, image_names, images):
        name = name.split('.')
        name = name[0] + f'-{label}.' + name[1]
        print(f"Output {name}...")
        image.save(os.path.join(out_path, name))

