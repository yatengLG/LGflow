# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor
from .module import Parameter, Module


class Dropout(Module):
    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(p=self.p, train=self.training)
