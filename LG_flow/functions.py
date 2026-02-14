# -*- coding: utf-8 -*-
# @Author  : LG


from .Math_op import Concat, Split, Dropout
from .tensor import Tensor


def concat(tensors: list, axis=0):
    assert isinstance(tensors, list)
    results = Concat(axis).forward(tensors)
    return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

def split(tensor: Tensor, split_size_or_sections, axis=0):
    assert isinstance(tensor, Tensor)
    results = Split(split_size_or_sections, axis).forward([tensor])
    return [Tensor(data=data, from_tensors=results.from_tensors, grad_fn=results.grad_fn, output_index=index)
                for index, data in enumerate(results.data)]

def dropout(tensor: Tensor, p, train=True):
    assert isinstance(tensor, Tensor)
    results = Dropout(p, train).forward([tensor])
    return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)


