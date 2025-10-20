# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor, randn, zeros
from .module import Parameter, Module


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()
