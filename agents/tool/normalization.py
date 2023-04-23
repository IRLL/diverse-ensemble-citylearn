# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 4:03 PM
# @Author  : Zhihu Yang
import numpy as np
from numba import jit


@jit
def _periodic_normalization_mul(x_max, x):
    x = 2 * np.pi * x / x_max
    x_sin = np.sin(x)
    x_cos = np.cos(x)
    return np.array([(x_sin + 1) / 2.0, (x_cos + 1) / 2.0])


@jit
def _onehot_encoding_mul(classes, x):
    identity_mat = np.eye(classes)
    return identity_mat[x]


@jit
def _normalize_mul(x_min, x_max, x):
    if x_min == x_max:
        return 0
    else:
        return (x - x_min) / (x_max - x_min)


class periodic_normalization:
    def __init__(self, x_max):
        self.x_max = x_max

    def __mul__(self, x):
        return _periodic_normalization_mul(self.x_max, x)

    def __rmul__(self, x):
        return _periodic_normalization_mul(self.x_max, x)


class onehot_encoding:
    def __init__(self, classes):
        self.classes = classes

    def __mul__(self, x):
        return _onehot_encoding_mul(self.classes, x)

    def __rmul__(self, x):
        return _onehot_encoding_mul(self.classes, x)


class normalize:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __mul__(self, x):
        return _normalize_mul(self.x_min, self.x_max, x)

    def __rmul__(self, x):
        return _normalize_mul(self.x_min, self.x_max, x)
