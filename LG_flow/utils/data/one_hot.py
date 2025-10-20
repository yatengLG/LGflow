# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np


def to_one_hot(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]