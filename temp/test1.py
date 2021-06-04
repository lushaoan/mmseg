#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.13'
__copyright__ = 'Copyright 2021, PI'


import torch

res = torch.nn.functional.softmax(torch.tensor([13,9,9], dtype=torch.float32))
print(res)