# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor, randn, zeros
from LG_flow.nn.init import kaiming_uniform_
from .module import Parameter, Module
from ..functional import linear


class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weights = Parameter(randn(shape=(out_features, in_features), requires_grad=True))
        self.bias = Parameter(zeros(out_features,requires_grad=True)) if use_bias else None
        self.reset_parameters()

    def forward(self, x:Tensor):
        return linear(x, self.weights, self.bias)

    def reset_parameters(self):
        kaiming_uniform_(self.weights)

    def __str__(self):
        return "LG_flow.nn.Linear(in_features={}, out_features={}, use_bias={})".format(self.in_features, self.out_features, self.use_bias)
