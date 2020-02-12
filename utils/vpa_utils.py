#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/6
"""

import numpy as np


def norm_img(x):
    """
    正则化
    """
    x = x.astype('float')
    x /= 127.5
    x -= 1.
    return x


def avg_list(x_list):
    x_np = np.array(x_list)
    avg = np.average(x_np)
    return avg
