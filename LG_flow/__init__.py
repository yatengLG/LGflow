# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
from LG_flow import Math_op
import copy

float32 = np.float32
float64 = np.float64
int16 = np.int16
int32 = np.int64

class Tensor:
    def __init__(self, data, from_tensors=None, grad_fn=None, grad=None, dtype=float32, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data.astype(dtype)
        self.shape = self.data.shape
        self.dim = self.data.ndim
        self.dtype = self.data.dtype
        self.from_tensors = from_tensors
        self.grad_fn = grad_fn
        self.grad = grad
        self.requires_grad = requires_grad

    # 加法
    def __add__(self, other):
        if isinstance(other, Tensor):
            # 张量+ 张量
            results = Math_op.add_with_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        # 张量+ 常量
        results = Math_op.add_with_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 右加 常量+张量
    def __radd__(self, other):
        results = Math_op.add_with_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 减法,直接调用加法
    def __sub__(self, other):
        return self.__add__(-other)

    # 右减 常量-张量
    def __rsub__(self, other):
        return -self.__radd__(-other)

    # 乘
    def __mul__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.mul_with_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.mul_with_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 右乘
    def __rmul__(self, other):
        results = Math_op.mul_with_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 矩阵乘法
    def matmul(self, other):
        assert isinstance(other, Tensor)
        assert self.shape[-1] == other.shape[-2]
        results = Math_op.matmul.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 除
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.div_with_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.div_with_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 右除
    def __rtruediv__(self, other):
        results = Math_op.div_by_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # ==
    def __eq__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.eq_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.eq_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # !=
    def __ne__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.ne_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.ne_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # <
    def __lt__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.lt_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.lt_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # <=
    def __le__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.le_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.le_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # >
    def __gt__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.gt_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.gt_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # >=
    def __ge__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.ge_tensor.forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.ge_const.forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 次方
    def __pow__(self, power, modulo=None):
        results = Math_op.power.forward([self], exponents=power)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    def pow(self, power):
        return self.__pow__(power=power)

    # 取反,减法中调用
    def __neg__(self):
        results = Math_op.neg.forward([self])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    def neg(self):
        return self.__neg__()

    # 取绝对值
    def __abs__(self):
        results = Math_op.abs.forward([self])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    def abs(self):
        return self.__abs__()

    # 截断
    def clip(self,min=None, max=None):
        results = Math_op.clip.forward([self],min=min, max=max)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 索引单个元素
    def item(self, index):
        assert isinstance(index, int) or isinstance(index, tuple)
        results = Math_op.item.forward([self], index)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 索引单个元素并替换
    def itemset(self, index, value):
        """
        返回一个新的张量,与原张量不共享
        :param index:
        :param value:
        :return:
        """
        assert isinstance(index, int) or isinstance(index, tuple)
        results = Math_op.itemset.forward([self], index, value)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 求最小值
    def min(self, axis=None, keepdims=False):
        assert isinstance(axis, int) or (axis is None)
        results = Math_op.min.forward([self], axis, keepdims)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 求最大值
    def max(self,axis=None, keepdims=False):
        assert isinstance(axis, int) or (axis is None)
        results = Math_op.max.forward([self], axis, keepdims)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 求和
    def sum(self, axis=None, keepdims=False):
        assert isinstance(axis, int) or (axis is None)
        results = Math_op.sum.forward([self], axis, keepdims)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 求均值
    def mean(self, axis=None, keepdims=False):
        assert isinstance(axis, int) or (axis is None)
        results = Math_op.mean.forward([self], axis, keepdims)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 求方差
    def std(self, axis=None, keepdims=False):
        assert isinstance(axis, int) or (axis is None)
        results = Math_op.std.forward([self], axis, keepdims)
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    # 数据data原地改变后,更新属性
    def upattr(self):
        self.shape = self.data.shape
        self.dim = self.data.ndim
        self.dtype = self.data.dtype

    # 填充
    def fill_(self, value):
        self.data.fill(value)
        self.upattr()

    # int()
    def int16(self):
        return Tensor(data=self.data, from_tensors=self.from_tensors, grad_fn=self.grad_fn, grad=self.grad, dtype=int16, requires_grad=self.requires_grad)

    # int_()
    def int16_(self):
        self.data = self.data.astype(int16)
        self.upattr()

    # int()
    def int32(self):
        return Tensor(data=self.data, dtype=int32, from_tensors=self.from_tensors, grad_fn=self.grad_fn, grad=self.grad, requires_grad=self.requires_grad)

    # int_()
    def int32_(self):
        self.data = self.data.astype(int32)
        self.upattr()

    # float()
    def float32(self):
        return Tensor(data=self.data, dtype=float32, from_tensors=self.from_tensors, grad_fn=self.grad_fn, grad=self.grad, requires_grad=self.requires_grad)

    # float_()
    def float32_(self):
        self.data = self.data.astype(float32)
        self.upattr()

    # float()
    def float64(self):
        return Tensor(data=self.data, dtype=float64, from_tensors=self.from_tensors, grad_fn=self.grad_fn, grad=self.grad, requires_grad=self.requires_grad)

    # float_()
    def float64_(self):
        self.data = self.data.astype(float64)
        self.upattr()
    # 复制
    def copy(self):
        return copy.deepcopy(self)
    #
    def required_grad(self):
        self.requires_grad = True
        return self
    #
    def no_grad(self):
        self.requires_grad = True
    # 打印使用
    def __str__(self):
        return "(Tensor shape={} dtype={} required_grad={} grad_fn={} \n{}\n)".format(self.shape, self.dtype, self.requires_grad, self.grad_fn, self.data)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones(self.shape)
        else:
            grad = grad
        if not self.requires_grad:
            self.grad = None

        if self.grad_fn is not None:
            grads = self.grad_fn.backward(self.from_tensors, grad)
            for tensor, grad in zip(self.from_tensors, grads):

                if isinstance(tensor, Tensor):

                    if tensor.grad is not None and tensor.grad_fn is None:
                        # print("存在 更新分支")
                        # print(tensor.grad_fn)
                        # print("update: {}".format(update))
                        # old_grad = tensor.grad
                        tensor.grad = tensor.grad + grad
                        # print("tensor.grad = tensor.grad + grad\n{}\n = {} \n+ {}".format(tensor.grad, old_grad, grad))
                        tensor.backward(tensor.grad)
                    else:
                        # print("不存在")
                        # print(tensor.grad_fn)
                        # print("tensor.grad =  grad\n{}".format(grad))

                        tensor.grad = grad
                        tensor.backward(tensor.grad)

class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data.data
        Tensor.__init__(self,data, requires_grad=True)

    def __str__(self):
        return "(Parameter shape={} dtype={} required_grad={} grad_fn={} \n{}\n)".format(self.shape, self.dtype, self.requires_grad, self.grad_fn, self.data)

# 创建全1张量
def ones(shape, dtype=float32, requires_grad=False):
    return Tensor(np.ones(shape=shape), dtype=dtype, requires_grad=requires_grad)

# 创建全0张量
def zeros(shape, dtype=float32, requires_grad=False):
    return Tensor(np.zeros(shape=shape), dtype=dtype, requires_grad=requires_grad)

# 从numpy 创建张量
def from_numpy(ndarry, dtype=float32, requires_grad=False):
    return Tensor(ndarry, dtype=dtype, requires_grad=False)

# 正太分布创建张量
def randn(shape, dtype=float32, requires_grad=False):
    return Tensor(np.random.normal(loc=0,scale=1,size=shape), dtype=dtype, requires_grad=requires_grad)

def randint(low, high, shape, dtype=int16, requires_grad=False):
    return Tensor(np.random.randint(low=low, high=high, size=shape), dtype=dtype, requires_grad=requires_grad)


