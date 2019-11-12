# -*- coding: utf-8 -*-
# @Author  : LG

import torch
import numpy as np

class Tensor:
    def __init__(self, data, from_tensors=None, op=None, grad=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.shape = data.shape
        self.dim = data.ndim
        self.from_tensors = from_tensors
        self.op = op
        self.grad = grad
    def t(self):
        return self.data.T
    def __add__(self, other):
        return add.forward([self, other])

    def sum(self,dim=None):
        return sum.forward([self], dim)

    def __mul__(self, other):
        return mul.forward([self, other])

    def mul(self, other):
        return mul.forward([self, other])

    def matmul(self, other):
        return matmul.forward([self, other])

    def __str__(self):
        return "{}({},shape:{})".format(self.data, self.__class__.__name__, self.shape)


    def backward(self, grad=None):
        if grad is None:
            self.grad = grad = np.ones(self.shape)
        else:
            self.grad = grad
        if self.op is not None:
            grads = self.op.backward(self.from_tensors, grad)
            for tensor, grad in zip(self.from_tensors, grads):
                tensor.grad = grad
                tensor.backward(tensor.grad)


class OP:
    def forward(self, from_tensors):
        raise NotImplementedError

    def backward(self, from_tensors, grad):
        raise NotImplementedError


class ADD(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data + from_tensors[1].data, from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [grad, grad]


class SUM(OP):
    def forward(self, from_tensors, dim=None):
        return Tensor(data=np.sum(from_tensors[0].data, axis=dim), from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [np.ones(shape=from_tensors[0].shape)*grad]


class MUL(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data*from_tensors[1].data, from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad, from_tensors[0].data * grad]


class MATMUL(OP):
    def forward(self, from_tensors):
        return Tensor(np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [np.matmul(grad, from_tensors[1].data.T), np.matmul(from_tensors[0].data.T, grad)]


add = ADD()
mul = MUL()
matmul = MATMUL()
sum=SUM()
a = Tensor([[1.,2.,3],[4,5,6]])
b = a.sum(dim=0)
c = b.sum(dim=None)


# print(a)

# b = Tensor([[1.,3.,2],[1,2,4]])
# print(b)
# print(a*b)
# c = Tensor([[1.,2.],[1.,2.],[3,4]])
# # print(c)
# e = a+b
# d = (e).matmul(c)
# print(d)
#
# d.backward()
#
# print("d.grad:",d.grad)
# print("e.grad:",e.grad)
# print("c.grad:",c.grad)
# print("b.grad:",b.grad)
# print("a.grad:",a.grad)
