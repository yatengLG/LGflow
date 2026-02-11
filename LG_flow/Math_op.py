# -*- coding: utf-8 -*-
# @Author  : LG

from collections import namedtuple
import numpy as np

results = namedtuple("numetuple_for_tensor", ["data", "from_tensors", "grad_fn"])


class Math(object):
    def forward(self, from_tensors):
        raise NotImplementedError

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 张量+常数
class ADD_WITH_CONST(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 2
        return results(from_tensors[0].data + from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad, None]


# 张量+张量
class ADD_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 2
        return results(from_tensors[0].data + from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        def reduce_grad(grad, target_shape):
            if grad.shape == target_shape:
                return grad

            grad_shape = grad.shape
            axes = []   # 需要求和的轴

            # 从后往前处理维度
            max_dims = max(len(grad_shape), len(target_shape))
            for i in range(1, max_dims + 1):
                try:
                    if target_shape[-i] == 1 and grad_shape[-i] > 1:
                        axes.append(max_dims - i)
                except:
                    axes.append(max_dims - i)

            if axes:
                grad = np.sum(grad, axis=tuple(axes), keepdims=True)
            return grad.reshape(target_shape)

        return [reduce_grad(grad, from_tensors[0].data.shape),
                reduce_grad(grad, from_tensors[1].data.shape)]


# 张量*常数
class MUL_WITH_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data * from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * from_tensors[1], None]


# 张量*张量
class MUL_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data * from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        data0 = from_tensors[0].data
        data1 = from_tensors[1].data
        grad0 = grad * data1
        grad1 = grad * data0

        def reduce_grad(grad, target_shape):
            if grad.shape == target_shape:
                return grad

            grad_shape = grad.shape
            axes = []   # 需要求和的轴

            # 从后往前处理维度
            max_dims = max(len(grad_shape), len(target_shape))
            for i in range(1, max_dims + 1):
                try:
                    if target_shape[-i] == 1 and grad_shape[-i] > 1:
                        axes.append(max_dims - i)
                except:
                    axes.append(max_dims - i)
            if axes:
                grad = np.sum(grad, axis=tuple(axes), keepdims=True)
            return grad.reshape(target_shape)
        grad0 = reduce_grad(grad0, data0.shape)
        grad1 = reduce_grad(grad1, data1.shape)
        return [grad0, grad1]


# 张量/常量
class DIV_WITH_CONST:
    def forward(self, from_tensors):
        return results(from_tensors[0].data / from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        data1 = from_tensors[1]
        grad0 = grad / data1
        return [grad0, None]


# 张量/张量
class DIV_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data / from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        data0 = from_tensors[0].data
        data1 = from_tensors[1].data
        grad0 = grad / data1
        grad1 = -grad * data0 / (data1 ** 2)

        def reduce_grad(grad, target_shape):
            if grad.shape == target_shape:
                return grad

            grad_shape = grad.shape
            axes = []

            # 从后往前处理维度
            max_dims = max(len(grad_shape), len(target_shape))
            for i in range(1, max_dims + 1):
                try:
                    if target_shape[-i] == 1 and grad_shape[-i] > 1:
                        axes.append(max_dims - i)
                except:
                    axes.append(max_dims - i)

            if axes:
                grad = np.sum(grad, axis=tuple(axes), keepdims=True)
            return grad.reshape(target_shape)

        grad0 = reduce_grad(grad0, data0.shape)
        grad1 = reduce_grad(grad1, data1.shape)
        return [grad0, grad1]


# 常量/张量
class DIV_BY_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[1] / from_tensors[0].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# == 张量
class EQ_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data == from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# == 张量
class EQ_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data == from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# != 张量
class NE_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data != from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# != 常量
class NE_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data != from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# <
class LT_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data < from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# <
class LT_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data < from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# <=
class LE_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data <= from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# <=
class LE_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data <= from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# >
class GT_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data > from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# >
class GT_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data > from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# >=
class GE_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data >= from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# >=
class GE_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data >= from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 指数次方
class POWER(Math):
    def __init__(self, exponents: int):
        self.exponents = exponents

    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            np.power(from_tensors[0].data, self.exponents), from_tensors, self
        )

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 取负
class NEG(Math):
    def forward(self, from_tensors):
        return results(-from_tensors[0].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [-grad]


# 取正
class POS(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 截断
class CLIP(Math):
    def __init__(self, min, max):
        assert any([min is not None, max is not None])
        self.min = min
        self.max = max

    def forward(self, from_tensors):
        return results(
            from_tensors[0].data.clip(min=self.min, max=self.max), from_tensors, self
        )

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 索引单个元素
class ITEM(Math):
    def forward(self, from_tensors, *args):
        return results(from_tensors[0].data.item(*args), from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 索引单个元素并替换
class ITEMSET(Math):
    def forward(self, from_tensors, *args):
        b = from_tensors[0].data.copy()
        b.itemset(*args)
        return results(b, from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 最大值, 维度
class MAX(Math):
    def __init__(self, axis: int = None, keepdims: bool = False):
        self.axis = axis
        self.keepdims = keepdims
        self.indices = None

    def forward(self, from_tensors):
        assert len(from_tensors) == 1

        self.indices = np.argmax(from_tensors[0].data, axis=self.axis)
        return results(
            from_tensors[0].data.max(axis=self.axis, keepdims=self.keepdims),
            from_tensors,
            self,
        )

    def backward(self, from_tensors, grad):
        data = from_tensors[0].data
        new_grad = np.zeros_like(data)
        if self.axis is None:
            indices = np.unravel_index(np.argmax(data), data.shape)
            new_grad[indices] = grad
        else:
            indices = np.argmax(data, axis=self.axis, keepdims=True)
            np.put_along_axis(new_grad, indices, grad, axis=self.axis)
        return [new_grad]


# 最小值, 维度
class MIN(Math):
    def __init__(self, axis: int, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            from_tensors[0].data.min(axis=self.axis, keepdims=self.keepdims),
            from_tensors,
            self,
        )

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 绝对值
class ABS(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(np.abs(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 和, 维度
class SUM(Math):
    def __init__(self, axis: int, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            from_tensors[0].data.sum(axis=self.axis, keepdims=self.keepdims),
            from_tensors,
            self,
        )

    def backward(self, from_tensors, grad):
        data = from_tensors[0].data
        if self.keepdims:
            new_grad = np.broadcast_to(grad, data.shape)
        else:
            new_grad = np.zeros_like(data)
            if self.axis is None:
                new_grad[:] = grad
            else:
                indices = [slice(None)] * len(data.shape)

                if isinstance(self.axis, tuple):
                    for ax in self.axis:
                        indices[ax] = np.newaxis
                else:
                    indices[self.axis] = np.newaxis

                expanded_grad = grad[tuple(indices)]
                new_grad = np.broadcast_to(expanded_grad, data.shape)
        return [new_grad]


# 均值 维度
class MEAN(Math):
    def __init__(self, axis: int, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            from_tensors[0].data.mean(axis=self.axis, keepdims=self.keepdims),
            from_tensors,
            self,
        )

    def backward(self, from_tensors, grad):
        raise NotImplementedError


# 方差 维度
class STD(Math):
    def __init__(self, axis: int, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            from_tensors[0].data.std(axis=self.axis, keepdims=self.keepdims),
            from_tensors,
            self,
        )

    def backward(self, from_tensors, grad):
        raise NotImplementedError


class EXP(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(
            np.exp(from_tensors[0].data), from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [grad * np.exp(from_tensors[0].data)]


class LOG(Math):
    """
    与torch的计算结果存在些微差异
    """
    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(np.log(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[0].data]


class T(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(from_tensors[0].data.T, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad.T]


class ReLU(Math):
    def forward(self, from_tensors):
        assert len(from_tensors) == 1
        return results(np.maximum(0, from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * (from_tensors[0].data > 0)]


# 矩阵乘法
class MATMUL(Math):
    def forward(self, from_tensors):
        return results(
            np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [
            np.matmul(grad, from_tensors[1].data.T),
            np.matmul(from_tensors[0].data.T, grad),
        ]
