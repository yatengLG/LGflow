# -*- coding: utf-8 -*-
# @Author  : LG

from tensor import Tensor
from nn.mudule import Linear

input = Tensor(data=([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
                     [1,1,1,1,1,1,2,3,4,5,1,2,3,4,5]))
fc1 = Linear(in_features=15, out_features=5)
fc2 = Linear(in_features=5, out_features=3)

out1 = fc1.forward(input)
out2 = fc2.forward(out1)

print(out2)
print(out1)

out2.backward()

print(fc2.weight.grad)
print(fc2.bias.grad)
print(fc1.weight.grad)
print(fc1.bias.grad)
