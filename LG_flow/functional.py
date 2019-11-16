# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow import Tensor

def linear(input_tensor:Tensor, weight:Tensor, bias:Tensor=None)->Tensor:
    output = input_tensor.matmul(weight)
    if bias is not None:
        output = output+bias
    return output




