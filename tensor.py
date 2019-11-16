# -*- coding: utf-8 -*-
# @Author  : LG

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

    # 转置
    def t(self):
        return t.forward([self])
    # e**self
    def exp(self):
        return exp.forward([self])
    # log(self)
    def log(self):
        return log.forward([self])
    # 加法
    def __add__(self, other):
        return add.forward([self, other])
    # 求和, 可指定维度
    def sum(self,dim=None):
        return sum.forward([self], dim)
    # 乘法
    def __mul__(self, other):
        return mul.forward([self, other])

    def mul(self, other):
        return mul.forward([self, other])

    def matmul(self, other):
        return matmul.forward([self, other])

    def matdiv(self, other):
        return matdiv.forward([self, other])

    def __str__(self):
        return "{}\n\t({} shape={} grad_fn={})".format(self.data, self.__class__.__name__, self.shape, self.op)

    def softmax(self,axis=1):
        return softmax.forward([self], axis=axis)

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
        self.dim = dim
        data = np.sum(from_tensors[0].data, axis=dim)
        return Tensor(data=data, from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        shape = from_tensors[0].shape
        new_shape = list(shape)
        new_shape[self.dim] = 1

        grad = grad.reshape(new_shape)
        return [np.repeat(grad, repeats=shape[self.dim], axis=self.dim)]


class MUL(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data*from_tensors[1].data, from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad, from_tensors[0].data * grad]


class MATMUL(OP):
    def forward(self, from_tensors):
        assert from_tensors[0].shape==from_tensors[1].shape
        return Tensor(np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [np.matmul(grad, from_tensors[1].data.T), np.matmul(from_tensors[0].data.T, grad)]

class MATDIV(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data / from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[1].data,
                -grad * from_tensors[0].data / (from_tensors[1].data * from_tensors[1].data)]


class EXP(OP):
    def forward(self, from_tensors):
        return Tensor(np.exp(from_tensors[0].data), from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):

        return [grad * np.exp(from_tensors[0].data)]


class LOG(OP):
    def forward(self, from_tensors):
        return Tensor(np.log(from_tensors[0].data), from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[0].data]


class T(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data.T, from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        return [grad.T]


class SOFTMAX(OP):
    def forward(self, from_tensors, axis=1):
        assert from_tensors[0].dim ==2
        a_exp = np.exp(from_tensors[0].data)
        sum_ = np.sum(a_exp, axis=axis)
        if axis==1:
            return Tensor((a_exp.T / sum_).T, from_tensors=from_tensors, op=self)
        elif axis==0:
            return Tensor(a_exp /sum_,from_tensors=from_tensors, op=self)

    def backward(self, from_tensors, grad):
        pass


add = ADD()
mul = MUL()
matmul = MATMUL()
matdiv = MATDIV()
sum = SUM()
exp = EXP()
log = LOG()
t = T()
softmax = SOFTMAX()