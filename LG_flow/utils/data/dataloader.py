# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import random


class DataLoader(object):
    def __init__(self, dataset, batch=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indexes = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indexes)

        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        indexes = self.indexes[self.i:self.i + self.batch]

        images = []
        labels = []
        for index in indexes:
            images.append(self.dataset.data[index])
            labels.append(self.dataset.labels[index])

        if len(images) < 1:
            self.i = 0
            if self.shuffle:
                random.shuffle(self.indexes)
            raise StopIteration

        elif self.drop_last and len(images) < self.batch:
            self.i = 0
            if self.shuffle:
                random.shuffle(self.indexes)
            raise StopIteration

        else:
            images = np.array(images)
            labels = np.array(labels)

            self.i += self.batch

            return images, labels

    def __len__(self):
        return len(self.dataset)
