# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow import Tensor

def linear(input_tensor:Tensor, weight:Tensor, bias:Tensor=None)->Tensor:
    output = input_tensor.matmul(weight)
    if bias is not None:
        output = output+bias
    return output

def softmax(input: Tensor, axis:int=1)->Tensor:
    x_max = input.max(axis=axis, keepdims=True)
    x_std = input - x_max
    x_exp = x_std.exp()
    output = x_exp / x_exp.sum(axis=axis, keepdims=True)
    return output

def cross_entropy(input:Tensor, target: Tensor, reduction: str = "mean")->Tensor:
    x_sortmax = softmax(input, axis=1)
    x_log = x_sortmax.log()
    output = - target * x_log
    output = output.sum(axis=1)
    if reduction == "mean":
        output = output.mean()
    elif reduction == "sum":
        output = output.sum()
    elif reduction == "none":
        output = output
    else:
        raise ValueError("Invalid reduction option")
    return output



