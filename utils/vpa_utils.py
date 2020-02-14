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


def sigmoid_thr(val, thr, gap):
    """
    数值归一化
    """
    x = val - thr
    x = x / gap
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 3)  # 保留3位
