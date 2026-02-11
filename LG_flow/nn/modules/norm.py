# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor, randn, zeros, ones
from .module import Parameter, Module
from ..functional import layer_norm


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, bias: bool= True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weights = Parameter(ones(self.normalized_shape, requires_grad=True))
        self.bias = Parameter(zeros(self.normalized_shape, requires_grad=True)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weights, self.bias, self.eps)