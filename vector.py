from __future__ import division

import math
import numpy as np

def squared_error(x, y):
    err = np.subtract(x, y)
    squared_err = np.dot(err, err)
    return squared_err

def unit_vector(x):
    return x / vlength(x)

def real_part(x):
    u = [v.real for v in x]
    return np.array(u)

def normalize(x):
    return x / float(sum(x))

def vlength(x):
    length = math.sqrt(np.dot(x, x))
    return length

def cosine(x, y):
    xlen = vlength(x)
    ylen = vlength(y)
    return np.dot(x, y) / (xlen * ylen)

def kronecker_delta(x):
    y = []
    for v in x:
        y.append(x == v)
    return np.array(y)

def row_vector(M, index):
    x = M[index, :]
    return x

def column_vector(M, index):
    x = M[:, index]
    return x