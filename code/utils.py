import torch

def to_homogeneous(x):

    if x.shape[1] == 4:
        return x

    return torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)
