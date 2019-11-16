# -*- coding: utf-8 -*-
# @Author  : LG

import LG_flow
from LG_flow import nn


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(5,3, bias=False)
        self.fc2 = nn.Linear(3,2)

    def forward(self, x):

        x1 = self.fc1.forward(x)
        x2 = self.fc2.forward(x1)

        return x2

x = LG_flow.randint(0,10,(2,5)).float32()

model = net()

# print(model.parameters())


# print(model)


out = model.forward(x)


out.backward()

print(model.fc2.weights)
print(model.fc1.weights.grad)

# = LG_flow.Parameter(LG_flow.ones(model.fc2.weights.shape))
out = model.forward(x)

out.backward()

print(model.fc2.weights)
print(model.fc1.weights.grad)

# model.print_detail()