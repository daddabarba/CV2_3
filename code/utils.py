import torch
import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

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
    """
    Given an NxM image, where N is a number of points, normalizes each axis of the set of points (to be between 0 and 1)

    Parameters:
        t (torch.Tensor) : an NxM (e.g. Nx2) tensor of N (M dimensional) points
    """

    min_t = torch.min(t, dim=0)[0][None]
    max_t = torch.max(t, dim=0)[0][None]

    return (t-min_t)/(max_t-min_t)

def plot_status(pred, target_lmks, title):
    """
    Compares target with predicted landmarks

    Parameters:
        pred (torch.Tensor) : Nx2 matrix of predicted landmarks
        target_lmks (torch.Tensor) : Nx2 matrix of target landmarks
        title (str) : title of the plot
    """

    plt.figure()

    plt.scatter(pred[:, 0], pred[:, 1], label = "prediction", color = "b")
    plt.scatter(target_lmks[:, 0], target_lmks[:, 1], label = "target", color = "r")

    plt.title(title)
    plt.legend()
    plt.axis('equal')

    plt.show()




