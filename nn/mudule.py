# -*- coding: utf-8 -*-
# @Author  : LG

from tensor import Tensor
import numpy as np
from nn.functional import *


class Module(object):
    def __init__(self):
        pass

    def forward(self, *inputs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = Tensor(data=np.ones(shape=(in_features, out_features)))
        if bias:
            self.bias = Tensor(data=np.ones(shape=(out_features)))
        else:
            self.bias = None

    def forward(self, input):
        return linear(input,self.weight,self.bias)

