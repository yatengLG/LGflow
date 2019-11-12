# -*- coding: utf-8 -*-
# @Author  : LG

from tensor import Tensor

def linear(input:Tensor, weight:Tensor, bias:Tensor=None):

    if input.dim == 2 and bias is not None:
        ret = input.matmul(weight)+bias
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret

