import torch
import numpy as np

from PIL import Image

def to_homogeneous(x):
    """
    Adds homogeneous coordinate to input if needed

    Parameters:
        x (torch.Tensor) : Nx3 or Nx4 tensor where N is the number of points. If Nx3 a fourth column of ones is added
    """

    if x.shape[1] == 4:
        return x

    return torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)

def get_landmarks(path : str):
    """
    Returns landmark points array from file

    Parameters:
        path (str) : path to landmark points file
    """

    with open(path, "rt") as f:
        return np.array([int(line) for line in f.readlines()])

def im2np(path : str):
    """
    Converts a given image to numpy array

    Parameters:
        path (str) : path to image file
    """

    return np.asarray(
        Image.open(path)
    )[:, :, :3]

def torch_norm(t):

    min_t = torch.min(t, dim=0)[0][None]
    max_t = torch.max(t, dim=0)[0][None]

    return (t-min_t)/(max_t-min_t)



