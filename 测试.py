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

    def __add__(self, other):
        return add.forward([self, other])

    def __mul__(self, other):
        return mul.forward([self, other])

    def matmul(self, other):
        return matmul.forward([self, other])

    def __str__(self):
        return "{}({},shape:{})".format(self.data, self.__class__.__name__, self.shape)


    def backward(self, grad=None):
        if grad is None:
            self.grad = grad = np.ones(self.shape)
        #
        if self.op is not None:
            self.grad = self.op.backward(self.from_tensors, grad)




class OP:
    def forward(self, from_tensors):
        pass
    def backward(self, from_tensors, grad):
        pass

class ADD(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data + from_tensors[1].data)

    def backward(self, from_tensors, grad):
        return [grad, grad]

class MUL(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data*from_tensors[1].data, from_tensors=from_tensors, ops=self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad, from_tensors[0].data * grad]

class MATMUL(OP):
    def forward(self, from_tensors):
        return Tensor(np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors=from_tensors, ops=self)
    def backward(self, from_tensors, grad):
        return [np.matmul(grad, from_tensors[1].data.T), np.matmul(from_tensors[0].data.T, grad)]


add = ADD()
mul = MUL()
matmul = MATMUL()

a = Tensor([[1.,2.,3],[4,5,6]])
print(a)

b = Tensor([[1.,2.],[1.,2.],[3,4]])
print(b)

c = a.matmul(b)
print(c)

c.backward()
print(c.grad)

print(b.grad)
print(a.grad)


