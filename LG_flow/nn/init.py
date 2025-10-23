# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor
import math
import numpy as np


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode='fan_in'):
    assert tensor.data.ndim == 2, "Only support 2-D tensor"
    fan = tensor.data.shape[1] if mode == 'fan_in' else tensor.data.shape[0]
    gain = math.sqrt(2.0 / (1 + a ** 2)) # for relu or leaky relu
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    tensor.data = np.random.uniform(low=-bound, high=bound, size=tensor.data.shape)

