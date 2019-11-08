# -*- coding: utf-8 -*-
# @Author  : LG

import torch

a = torch.tensor([[1.,2.,3],[4,5,6]])
print(a)

b = torch.tensor([[1.,2.,3],[4,5,6]]).requires_grad
print(b)

c = a*b
print(c)

print(c.grad)