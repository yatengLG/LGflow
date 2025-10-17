# -*- coding: utf-8 -*-
# @Author  : LG

from LG_flow.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data.data
        Tensor.__init__(self,data, requires_grad=True)

    def __str__(self):
        return "(Parameter shape={} dtype={} required_grad={} grad_fn={} \n{}\n)".format(self.shape, self.dtype, self.requires_grad, self.grad_fn, self.data)

