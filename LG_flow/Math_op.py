# -*- coding: utf-8 -*-
# @Author  : LG

from collections import namedtuple
import numpy as np

results = namedtuple('numetuple_for_tensor',['data','from_tensors','grad_fn'])

class Math(object):
    def forward(self, from_tensors):
        raise NotImplementedError
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 张量+常数
class ADD_WITH_CONST(Math):
    def forward(self, from_tensors):
        assert len(from_tensors)==2
        return results(from_tensors[0].data + from_tensors[1], from_tensors, self)
    def backward(self, from_tensors, grad):
        return [grad, None]

# 张量+张量
class ADD_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        assert len(from_tensors)==2
        return results(from_tensors[0].data + from_tensors[1].data, from_tensors, self)
    def backward(self, from_tensors, grad):
        return [grad, grad]

# 张量*常数
class MUL_WITH_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data*from_tensors[1], from_tensors, self)
    def backward(self, from_tensors, grad):
        return [grad*from_tensors[1], None]

# 张量*张量
class MUL_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data*from_tensors[1].data, from_tensors, self)
    def backward(self, from_tensors, grad):
        return [grad * from_tensors[1].data, grad * from_tensors[0].data]

# 张量/常量
class DIV_WITH_CONST():
    def forward(self, from_tensors):
        return results(from_tensors[0].data/from_tensors[1], from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 张量*张量
class DIV_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data/from_tensors[1].data, from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 常量/张量
class DIV_BY_CONST(Math):
    def forward(self, from_tensors):
        return results(from_tensors[1]/from_tensors[0].data, from_tensors, self)
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
    def forward(self, from_tensors, exponents=2):
        assert len(from_tensors)==1
        assert isinstance(exponents, int)
        return results(np.power(from_tensors[0].data,exponents),from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError
# 取负
class NEG(Math):
    def forward(self, from_tensors):
        return results(-from_tensors[0].data, from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 取正
class POS(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data, from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 截断
class CLIP(Math):
    def forward(self, from_tensors, min=None, max=None):
        assert any([min is not None,max is not None])
        return results(from_tensors[0].data.clip(min=min,max=max), from_tensors, self)
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

# 切片

# 最大值, 维度
class MAX(Math):
    def forward(self, from_tensors, axis=None, keepdims=False):
        assert len(from_tensors)==1
        return results(from_tensors[0].data.max(axis=axis, keepdims=keepdims), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 最小值, 维度
class MIN(Math):
    def forward(self, from_tensors, axis=None, keepdims=False):
        assert len(from_tensors)==1
        return results(from_tensors[0].data.min(axis=axis, keepdims=keepdims), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 绝对值
class ABS(Math):
    def forward(self, from_tensors):
        assert len(from_tensors)==1
        return results(np.abs(from_tensors[0].data), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 和, 维度
class SUM(Math):
    def forward(self, from_tensors, axis=None, keepdims=False):
        assert len(from_tensors)==1
        return results(from_tensors[0].data.sum(axis=axis, keepdims=keepdims), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 均值 维度
class MEAN(Math):
    def forward(self, from_tensors, axis=None, keepdims=False):
        assert len(from_tensors)==1
        return results(from_tensors[0].data.mean(axis=axis, keepdims=keepdims), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 方差 维度
class STD(Math):
    def forward(self, from_tensors, axis=None, keepdims=False):
        assert len(from_tensors)==1
        return results(from_tensors[0].data.std(axis=axis, keepdims=keepdims), from_tensors, self)
    def backward(self, from_tensors, grad):
        raise NotImplementedError

# 矩阵乘法
class MATMUL(Math):
    def forward(self, from_tensors):
        return results(np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors, self)
    def backward(self, from_tensors, grad):
        return [np.matmul(grad, from_tensors[1].data.T), np.matmul(from_tensors[0].data.T, grad)]


add_with_const = ADD_WITH_CONST()
add_with_tensor = ADD_WITH_TENSOR()

mul_with_const = MUL_WITH_CONST()
mul_with_tensor = MUL_WITH_TENSOR()

div_with_const = DIV_WITH_CONST()
div_with_tensor = DIV_WITH_TENSOR()
div_by_const = DIV_BY_CONST()

eq_tensor = EQ_TENSOR()
eq_const = EQ_CONST()
ne_tensor = NE_TENSOR()
ne_const = NE_TENSOR()
lt_tensor = LT_TENSOR()
lt_const = LT_CONST()
le_tensor = LE_TENSOR()
le_const = LE_CONST()
gt_tensor = GT_TENSOR()
gt_const = GT_CONST()
ge_tensor = GE_TENSOR()
ge_const = GE_CONST()

neg = NEG()
clip = CLIP()
item = ITEM()
itemset = ITEMSET()

max = MAX()
min = MIN()
abs = ABS()
mean = MEAN()
std = STD()
sum = SUM()
power = POWER()

matmul = MATMUL()