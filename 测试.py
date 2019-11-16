# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from torch.nn import functional as f
from torch import nn

fc = nn.Linear(5,3)

print(fc._parameters)
