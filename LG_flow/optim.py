# -*- coding: utf-8 -*-
# @Author  : LG


class SGD:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for k, w in self.params.items():
            if  w is not None and w.grad is not None:
                w.grad = None

    def step(self):
        for k, w in self.params.items():
            if w is not None:
                w.data =  w.data - w.grad * self.lr
