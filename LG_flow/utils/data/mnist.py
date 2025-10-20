# -*- coding: utf-8 -*-
# @Author  : LG


from .dataset import Dataset
import os
import numpy as np


class Mnist(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        self.data, self.labels = self._load_data()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"

        with open(os.path.join(self.root, image_file), "rb") as f:
            # 读取头文件16字节
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')

            # 读取数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, rows, cols)

        with open(os.path.join(self.root, label_file), "rb") as f:
            # 读取头文件8字节
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')

            # 读取数据
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels


