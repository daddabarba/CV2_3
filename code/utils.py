import numpy as np

def to_homogeneous(x):

    if x.shape[1] != 3:
        x = x.T

    return np.c_[x, np.ones((x.shape[0], 1))]
