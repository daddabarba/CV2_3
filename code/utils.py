import numpy as np

def to_homogeneous(x):

    if x.shape[1] == 4:
        return x

    return np.c_[x, np.ones((x.shape[0], 1))]
