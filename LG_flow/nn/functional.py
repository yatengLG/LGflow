# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor


def linear(input_tensor:Tensor, weight:Tensor, bias:Tensor=None)->Tensor:
    output = input_tensor.matmul(weight.T())
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

def layer_norm(input:Tensor, normalized_shape:tuple, weight:Tensor = None, bias:Tensor = None, eps=1e-5)->Tensor:
    dims = tuple(range(-len(normalized_shape), 0))

    mean = input.mean(axis=dims, keepdims=True)
    var = input.var(axis=dims, keepdims=True, unbiased=False)

    input_norm = (input - mean) / ((var + eps) ** 0.5)

    if weight is not None:
        input_norm = input_norm * weight

    if bias is not None:
        input_norm = input_norm + bias
    return input_norm
