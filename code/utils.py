import torch

from scipy import misc

def to_homogeneous(x):
    """
    Adds homogeneous coordinate to input if needed

    Parameters:
        x (torch.Tensor) : Nx3 or Nx4 tensor where N is the number of points. If Nx3 a fourth column of ones is added
    """

    if x.shape[1] == 4:
        return x

    return torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)

def im2np(path : str):
    """
    Converts a given image to numpy array

    Parameters:
        path (str) : path to image file
    """
    return misc.imread(path)
