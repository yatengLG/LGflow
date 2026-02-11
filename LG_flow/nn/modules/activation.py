# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor, randn, zeros
from .module import Parameter, Module
from LG_flow.nn.functional import softmax


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class Softmax(Module):
    def forward(self, input: Tensor, axis:int=1) -> Tensor:
        return softmax(input, axis)