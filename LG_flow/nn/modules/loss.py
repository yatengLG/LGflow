# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor
from .module import Module
from ..functional import cross_entropy


class CrossEntropyLoss(Module):
    def __init__(self, reduction='sum'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(input, target, self.reduction)